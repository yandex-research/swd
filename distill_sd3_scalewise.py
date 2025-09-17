#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import math
import os
from copy import deepcopy
from pathlib import Path

import random
import datasets
import torch
import transformers
import types
import diffusers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict, get_peft_model
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from diffusers.loaders import SD3LoraLoaderMixin
from diffusers.image_processor import VaeImageProcessor
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers

from src.utils.train_utils import log_validation, tokenize_captions, \
    encode_prompt, unwrap_model, tokenize_prompt, distributed_sampling
from src.utils.flow_matching_sampler import FlowMatchingSolver
from src.pipelines.stable_diffusion_3 import ScaleWiseStableDiffusion3Pipeline
from src.models.transformer_with_gan import forward_with_classify, TransformerCls
from src.utils.distillation_losses import dmd_loss, fake_diffusion_loss, pdm_loss_real, pdm_loss_fake
from src.utils.metrics import calculate_scores

logger = get_logger(__name__)

DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}


########################################################################################################################
#                                               TRAINING                                                               #
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
def train(args):
    ## Prepare accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = prepare_accelerator(args, logging_dir, True)
    accelerator_fake = prepare_accelerator(args, logging_dir, False)
    
    # Some useful asserts
    assert args.gradient_accumulation_steps == 1, "Grad accumulation is not implemented here"
    assert args.max_eval_samples % accelerator.num_processes == 0, \
        "Must be divisible by world size. Otherwise, allgather fails."

    torch.set_float32_matmul_precision('high')

    ## Prepare models
    transformer, transformer_teacher, transformer_fake, vae, \
    text_encoder, text_encoder_2, text_encoder_3, \
    tokenizer, tokenizer_2, tokenizer_3, noise_scheduler, weight_dtype = prepare_models(args, accelerator)

    ## Add CNN upscaling and make it DDP
    transformer = accelerator.prepare(transformer)

    ## Set up schedulers: diffusion and distilled models
    fm_solver = FlowMatchingSolver(noise_scheduler, args.num_boundaries, args.scales, args.boundaries)

    ## Add GAN head and make it DDP
    transformer_fake.forward = types.MethodType(forward_with_classify, transformer_fake)
    if args.pdm_real:
        transformer_teacher.forward = types.MethodType(forward_with_classify, transformer_teacher)
    
    transformer_fake = TransformerCls(args, transformer_fake)
    transformer_fake = accelerator_fake.prepare(transformer_fake)

    ## Prepare optimizers
    optimizer, params_to_optimize = prepare_optimizer(args, transformer, is_student=True)
    optimizer = accelerator.prepare(optimizer)
    
    optimizer_fake, params_to_optimize_fake = prepare_optimizer(
        args, transformer_fake, is_student=False
    )
    optimizer_fake = accelerator_fake.prepare(optimizer_fake)

    ## Prepare data
    train_dataloader = # <YOUR CODE>: init dataloader with args.train_batch_size
    train_dataset = train_dataloader.dataset

    ## Load if exist
    initial_global_step = load_if_exist(args, accelerator, len(train_dataloader))

    ## Prepare 3rd party utils
    prepare_3rd_party(args, accelerator, len(train_dataloader))
    ## -----------------------------------------------------------------------------------------------

    ## TRAINING STAGE
    ## -----------------------------------------------------------------------------------------------
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    if hasattr(train_dataset, "__len__"):
        logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    uncond_prompt_embeds, uncond_pooled_prompt_embeds = prepare_prompt_embed_from_caption(
        [''] * args.train_batch_size, tokenizer, tokenizer_2, tokenizer_3,
        text_encoder, text_encoder_2, text_encoder_3
    )

    assert transformer.training
    for batch in train_dataloader:

        idx_start = random.choices(fm_solver.boundary_start_idx, k=1)[0]
        idx_start = torch.tensor([idx_start] * args.train_batch_size).long()
              
        ### DMD loss
        ### ----------------------------------------------------
        if args.do_dmd_loss:
            for _ in range(args.n_steps_fake_dmd):
                model_input, model_input_prev, batch, \
                prompt_embeds, pooled_prompt_embeds = sample_batch(
                    args, accelerator,
                    batch, fm_solver,
                    tokenizer, tokenizer_2, tokenizer_3,
                    text_encoder, text_encoder_2, text_encoder_3,
                    vae, weight_dtype, idx_start
                )
                timesteps_start = noise_scheduler.timesteps[idx_start].to(device=model_input.device)

                avg_dmd_fake_loss = fake_diffusion_loss(
                    transformer, transformer_fake,
                    prompt_embeds, pooled_prompt_embeds,
                    model_input, timesteps_start, idx_start,
                    optimizer_fake,
                    params_to_optimize_fake,
                    weight_dtype, noise_scheduler, fm_solver,
                    accelerator_fake, args, model_input_down=model_input_prev
                )

            avg_dmd_loss = dmd_loss(
                transformer, transformer_fake,
                transformer_teacher,
                prompt_embeds, pooled_prompt_embeds,
                uncond_prompt_embeds, uncond_pooled_prompt_embeds,
                model_input, timesteps_start, idx_start,
                optimizer, params_to_optimize,
                weight_dtype, noise_scheduler, fm_solver,
                accelerator, args, model_input_down=model_input_prev
            )
        ### ----------------------------------------------------

        ### DM loss
        ### ----------------------------------------------------
        if args.do_pdm_loss:
            model_input, model_input_prev, batch, \
            prompt_embeds, pooled_prompt_embeds = sample_batch(
                args, accelerator,
                batch, fm_solver,
                tokenizer, tokenizer_2, tokenizer_3,
                text_encoder, text_encoder_2, text_encoder_3,
                vae, weight_dtype, idx_start
            )
            timesteps_start = noise_scheduler.timesteps[idx_start].to(device=model_input.device)
            
            pdm_loss = pdm_loss_real if args.pdm_real else pdm_loss_fake
            avg_pdm_loss = pdm_loss(
                transformer,
                transformer_teacher if args.pdm_real else transformer_fake,
                prompt_embeds, pooled_prompt_embeds,
                model_input, timesteps_start, idx_start,
                optimizer, params_to_optimize,
                weight_dtype, noise_scheduler, fm_solver,
                accelerator, args, model_input_down=model_input_prev
            )
        ### ----------------------------------------------------

        progress_bar.update(1)
        global_step += 1

        ### Model evaluation
        ### ----------------------------------------------------
        if global_step % args.evaluation_steps == 0:
            image_processor = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor)
            pipeline = Pipeline(
                vae=vae,
                transformer=unwrap_model(transformer, accelerator),
                text_encoder=unwrap_model(text_encoder, accelerator),
                text_encoder_2=unwrap_model(text_encoder_2, accelerator),
                text_encoder_3=unwrap_model(text_encoder_3, accelerator),
                tokenizer=tokenizer, tokenizer_2=tokenizer_2, tokenizer_3=tokenizer_3,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
                image_processor=image_processor
            )

            for eval_set_name in ['mjhq', 'coco']:
                eval_prompts_path = f'prompts/{eval_set_name}.csv'
                if eval_set_name == "coco":
                    fid_stats_path = args.coco_ref_stats_path
                else:
                    fid_stats_path = args.mjhq_ref_stats_path

                images, prompts = distributed_sampling(
                    pipeline,
                    args,
                    eval_prompts_path,
                    prepare_prompt_embed_from_caption,
                    fm_solver,
                    noise_scheduler,
                    accelerator,
                    logger,
                )
                if accelerator.is_main_process:
                    torch.cuda.empty_cache()
                    image_reward, pick_score, clip_score, fid_score = calculate_scores(
                        args,
                        images,
                        prompts,
                        ref_stats_path=fid_stats_path,
                    )
                    logs = {
                        f"fid_{eval_set_name}": fid_score.item(),
                        f"pick_score_{eval_set_name}": pick_score.item(),
                        f"clip_score_{eval_set_name}": clip_score.item(),
                        f"image_reward_{eval_set_name}": image_reward.item(),
                    }
                    accelerator.log(logs, step=global_step)

                torch.cuda.empty_cache()
                accelerator.wait_for_everyone()
        ### ----------------------------------------------------

        ### Saving checkpoint
        if accelerator.is_main_process:
            if global_step % args.evaluation_steps == 0:
                saving(transformer, args, accelerator, global_step, logs)
        accelerator.wait_for_everyone()

        ### Log validation images
        ### ----------------------------------------------------
        if accelerator.is_main_process:
            if global_step % args.validation_steps == 0:
                image_processor = VaeImageProcessor(vae_scale_factor=vae.config.scaling_factor)
                pipeline = Pipeline(
                    vae=vae,
                    transformer=unwrap_model(transformer, accelerator),
                    text_encoder=unwrap_model(text_encoder, accelerator),
                    text_encoder_2=unwrap_model(text_encoder_2, accelerator),
                    text_encoder_3=unwrap_model(text_encoder_3, accelerator),
                    tokenizer=tokenizer, tokenizer_2=tokenizer_2, tokenizer_3=tokenizer_3,
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                    image_processor=image_processor
                )

                log_validation(
                    pipeline, args,
                    prepare_prompt_embed_from_caption, fm_solver, noise_scheduler,
                    accelerator, logger, global_step,
                )
                
                del pipeline
                torch.cuda.empty_cache()

        accelerator.wait_for_everyone()
        ### ----------------------------------------------------

        logs = {
            "fake_loss": avg_dmd_fake_loss.detach().item() if args.do_dmd_loss else 0,
            "dmd_loss": avg_dmd_loss.detach().item() if args.do_dmd_loss else 0,
            "pdm_loss": avg_pdm_loss.detach().item() if args.do_pdm_loss else 0,
        }
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break
    ## -----------------------------------------------------------------------------------------------

    accelerator.wait_for_everyone()
    accelerator.end_training()



########################################################################################################################
#                                           TRAINING HELPER FUNCTIONS                                                  #
########################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
def sample_batch(
    args, accelerator,
    batch, fm_solver,
    tokenizer, tokenizer_2, tokenizer_3,
    text_encoder, text_encoder_2, text_encoder_3,
    vae, weight_dtype, idx_start
):
    pixel_values_ = batch["image"].to(device=accelerator.device)
    
    scales = fm_solver.scales[torch.argmax((idx_start[:, None] == fm_solver.boundary_start_idx).int(),
                                           dim=1)]
    current_scale = scales[0].item()

    if args.do_pixels_downscale:
        if current_scale == fm_solver.min_scale:
            previous_scales = scales
        else:
            previous_scales = fm_solver._get_previous_scale(scales)
        pixel_values = fm_solver.downscale_to_current(pixel_values_, scales * 8)
        pixel_values_prev = fm_solver.downscale_to_current(pixel_values, previous_scales * 8)
        model_input_prev = vae.encode(pixel_values_prev.to(weight_dtype)).latent_dist.sample()
        model_input_prev = (model_input_prev - vae.config.shift_factor) * vae.config.scaling_factor
        assert model_input_prev.dtype == weight_dtype
    else:
        model_input_prev = None

    # Memory efficient VAE encoding
    model_inputs = []
    for pixel_value in pixel_values:
        model_input = vae.encode(pixel_value[None].to(weight_dtype)).latent_dist.sample()
        model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
        model_inputs.append(model_input)
    model_input = torch.cat(model_inputs, dim=0)
    assert model_input.dtype == weight_dtype

    batch.update(tokenize_captions(batch, args, tokenizer, tokenizer_2, tokenizer_3, is_train=True))
    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        text_encoder_3=text_encoder_3,
        input_ids=batch["input_ids"],
        input_ids_2=batch["input_ids_2"],
        input_ids_3=batch["input_ids_3"]
    )

    return model_input, model_input_prev, batch, prompt_embeds, pooled_prompt_embeds
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def prepare_accelerator(args, logging_dir, find_unused_parameters=False):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        step_scheduler_with_optimizer=False,
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
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    return accelerator


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def prepare_models(args, accelerator):
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", shift=args.scheduler_shift
    )
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision
    )
    text_encoder_3 = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision
    )
    tokenizer_3 = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_3", revision=args.revision
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
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
    transformer.to(accelerator.device, dtype=weight_dtype, memory_format=torch.channels_last)
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
        target_modules.extend(["timestep_embedder.linear_1", "timestep_embedder.linear_2"])
    if args.apply_lora_to_proj_out:
        target_modules.extend(["proj_out"])
    if args.apply_lora_to_embedders:
        target_modules.extend(["x_embedder", "context_embedder"])
    assert len(target_modules) > 0, "LoRA has to be applied to at least one type of projection."
    
    transformer_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha if args.lora_alpha else args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )

    transformer_fake = copy.deepcopy(transformer)
    transformer = get_peft_model(transformer, transformer_lora_config)
    transformer_teacher = copy.deepcopy(transformer) if args.pdm_real else transformer # copy teacher to avoid backward issues for pdm real
    transformer_fake = get_peft_model(transformer_fake, transformer_lora_config)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the transformer attn processor layers
            # or there are the transformer and text encoder attn layers
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(model, accelerator), type(unwrap_model(transformer, accelerator))):
                    transformer_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            ScaleWiseStableDiffusion3Pipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer, accelerator))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = SD3LoraLoaderMixin.lora_state_dict(input_dir)
        transformer_state_dict = lora_state_dict
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if weight_dtype == torch.float16:
            models = [transformer_]
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer_teacher.enable_gradient_checkpointing()
        transformer.enable_gradient_checkpointing()
        transformer_fake.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Make sure the trainable params are in float32.
    if weight_dtype == torch.float16:
        models = [transformer]
        cast_training_params(models, dtype=torch.float32)
        models = [transformer_fake]
        cast_training_params(models, dtype=torch.float32)

    return transformer, transformer_teacher, transformer_fake, vae, \
           text_encoder, text_encoder_2, text_encoder_3, \
           tokenizer, tokenizer_2, tokenizer_3, \
           noise_scheduler, weight_dtype


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def prepare_prompt_embed_from_caption(
        caption, tokenizer, tokenizer_2, tokenizer_3,
        text_encoder, text_encoder_2, text_encoder_3
):
    uncond_tokens = {"input_ids": tokenize_prompt(tokenizer, caption),
                     "input_ids_2": tokenize_prompt(tokenizer_2, caption),
                     "input_ids_3": tokenize_prompt(tokenizer_3, caption),
                     }
    uncond_prompt_embeds, uncond_pooled_prompt_embeds = encode_prompt(
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        text_encoder_3=text_encoder_3,
        input_ids=uncond_tokens["input_ids"],
        input_ids_2=uncond_tokens["input_ids_2"],
        input_ids_3=uncond_tokens["input_ids_3"],
    )
    return uncond_prompt_embeds, uncond_pooled_prompt_embeds


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def prepare_optimizer(args, transformer, is_student=True):
    ## Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate if is_student else args.learning_rate_cls,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    return optimizer, params_to_optimize


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def load_if_exist(args, accelerator):
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
    else:
        initial_global_step = 0

    return initial_global_step


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def prepare_3rd_party(args, accelerator, dataloader_size):
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(dataloader_size / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(dataloader_size / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(deepcopy(args))
        # Tracker config doesn't support lists
        tracker_config.pop("sizes")
        tracker_config.pop("size_switch_steps")
        tracker_config.pop("scales")
        tracker_config.pop("cls_blocks")
        tracker_config.pop("pdm_blocks")
        tracker_config.pop("boundaries")
        accelerator.init_trackers("scalewise", config=tracker_config)


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def saving(transformer, args, accelerator, global_step, logs):
    # Save by ps
    # condition = logs['pick_score_coco'] > args.previous_best_ps
    # if condition:
    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    print(f'Saving to {save_path}')
    transformer = accelerator.unwrap_model(transformer)
    transformer.save_pretrained(save_path)
    logger.info(f"Saved state to {save_path}")

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
class Pipeline:
    def __init__(self, vae, transformer,
                 text_encoder, text_encoder_2, text_encoder_3,
                 tokenizer, tokenizer_2, tokenizer_3,
                 revision, variant, torch_dtype, image_processor):
        self.vae = vae
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.text_encoder_3 = text_encoder_3
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3
        self.revision = revision
        self.variant = variant
        self.torch_dtype = torch_dtype
        self.image_processor = image_processor
# ----------------------------------------------------------------------------------------------------------------------
