import copy
import logging
import os

import diffusers
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
)
from diffusers.training_utils import cast_training_params
from peft import LoraConfig, get_peft_model
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

SD35_MODEL_NAMES = {
    "medium": "stabilityai/stable-diffusion-3.5-medium",
    "large": "stabilityai/stable-diffusion-3.5-large",
}

logger = get_logger(__name__)


# ----------------------------------------------------------------------------------------------------------------------
def prepare_models(args, accelerator):
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        SD35_MODEL_NAMES[args.model_name],
        subfolder="scheduler",
        shift=args.scheduler_shift,
    )
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        SD35_MODEL_NAMES[args.model_name],
        subfolder="text_encoder",
        revision=args.revision,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        SD35_MODEL_NAMES[args.model_name], subfolder="tokenizer", revision=args.revision
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        SD35_MODEL_NAMES[args.model_name],
        subfolder="text_encoder_2",
        revision=args.revision,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        SD35_MODEL_NAMES[args.model_name],
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    text_encoder_3 = T5EncoderModel.from_pretrained(
        SD35_MODEL_NAMES[args.model_name],
        subfolder="text_encoder_3",
        revision=args.revision,
    )
    tokenizer_3 = T5TokenizerFast.from_pretrained(
        SD35_MODEL_NAMES[args.model_name],
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    vae = AutoencoderKL.from_pretrained(
        SD35_MODEL_NAMES[args.model_name],
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        SD35_MODEL_NAMES[args.model_name],
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    text_encoder_3.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move transformer, vae and text_encoder to device and cast to weight_dtype
    transformer.to(
        accelerator.device, dtype=weight_dtype, memory_format=torch.channels_last
    )
    vae.to(accelerator.device, dtype=weight_dtype, memory_format=torch.channels_last)

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    text_encoder_3.to(accelerator.device, dtype=weight_dtype)

    # add new LoRA weights
    # Set correct lora layers
    # Default choice
    target_modules = []
    if args.apply_lora_to_attn_projections:
        target_modules.extend(["to_k", "to_q", "to_v", "to_out.0"])
    if args.apply_lora_to_add_projections:
        target_modules.extend(["add_k_proj", "add_q_proj", "add_v_proj", "to_add_out"])
    if args.apply_lora_to_mlp_projections:
        target_modules.extend(["net.0.proj", "net.2"])
    if args.apply_lora_to_ada_norm_projections:
        target_modules.extend(["norm1.linear", "norm1_context.linear"])
    if args.apply_lora_to_timestep_projections:
        target_modules.extend(
            ["timestep_embedder.linear_1", "timestep_embedder.linear_2"]
        )
    if args.apply_lora_to_proj_out:
        target_modules.extend(["proj_out"])
    if args.apply_lora_to_embedders:
        target_modules.extend(["x_embedder", "context_embedder"])
    assert (
        len(target_modules) > 0
    ), "LoRA has to be applied to at least one type of projection."

    transformer_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha if args.lora_alpha else args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )

    transformer_fake = copy.deepcopy(transformer)
    transformer = get_peft_model(transformer, transformer_lora_config)
    transformer_teacher = transformer
    transformer_fake = get_peft_model(transformer_fake, transformer_lora_config)

    if args.gradient_checkpointing:
        transformer_teacher.enable_gradient_checkpointing()
        transformer.enable_gradient_checkpointing()
        transformer_fake.enable_gradient_checkpointing()

    # Make sure the trainable params are in float32.
    if weight_dtype == torch.float16:
        models = [transformer]
        cast_training_params(models, dtype=torch.float32)
        models = [transformer_fake]
        cast_training_params(models, dtype=torch.float32)

    return (
        transformer,
        transformer_teacher,
        transformer_fake,
        vae,
        text_encoder,
        text_encoder_2,
        text_encoder_3,
        tokenizer,
        tokenizer_2,
        tokenizer_3,
        noise_scheduler,
        weight_dtype,
    )


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def prepare_optimizer(args, transformer, is_student=True):
    params_to_optimize = list(
        filter(lambda p: p.requires_grad, transformer.parameters())
    )

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate if is_student else args.learning_rate_cls,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    return optimizer, params_to_optimize


# ----------------------------------------------------------------------------------------------------------------------
def prepare_accelerator(args, logging_dir, find_unused_parameters=False):
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=find_unused_parameters
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    return accelerator


# ----------------------------------------------------------------------------------------------------------------------
