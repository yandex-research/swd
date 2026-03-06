import random

import numpy as np
import torch
from diffusers.utils.torch_utils import is_compiled_module
from transformers import CLIPTextModelWithProjection, T5EncoderModel

########################################################################################################################
#                                       UTILS FUNCTIONS FOR TRAIN                                                      #
########################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def encode_prompt(
    text_encoder: CLIPTextModelWithProjection,
    text_encoder_2: CLIPTextModelWithProjection,
    text_encoder_3: T5EncoderModel,
    input_ids: torch.Tensor,
    input_ids_2: torch.Tensor,
    input_ids_3: torch.Tensor,
    device="cuda",
):
    # Prepare CLIP prompt embeds
    prompt_embeds = text_encoder(input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]

    prompt_embeds_2 = text_encoder_2(input_ids_2.to(device), output_hidden_states=True)
    pooled_prompt_embeds_2 = prompt_embeds_2[0]
    prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]

    # Prepare T5 prompt embeds
    t5_prompt_embeds = text_encoder_3(input_ids_3.to(device))[0]

    # Join prompt embeds
    clip_prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds,
        (0, t5_prompt_embeds.shape[-1] - clip_prompt_embeds.shape[-1]),
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embeds], dim=-2)
    pooled_prompt_embeds = torch.cat(
        [pooled_prompt_embeds, pooled_prompt_embeds_2], dim=-1
    )
    return prompt_embeds, pooled_prompt_embeds


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def tokenize_captions(examples, tokenizer, tokenizer_2, tokenizer_3, is_train=True):
    captions = []
    for caption in examples["text"]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                "Captions should contain either strings or lists of strings."
            )
    input_ids = tokenize_prompt(tokenizer, captions)
    input_ids_2 = tokenize_prompt(tokenizer_2, captions)
    input_ids_3 = tokenize_prompt(tokenizer_3, captions)
    return {
        "input_ids": input_ids,
        "input_ids_2": input_ids_2,
        "input_ids_3": input_ids_3,
    }


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def sample_batch(
    args,
    accelerator,
    global_step,
    loader,
    fm_solver,
    tokenizer,
    tokenizer_2,
    tokenizer_3,
    text_encoder,
    text_encoder_2,
    text_encoder_3,
    vae,
    weight_dtype,
):
    batch = next(loader)

    # Sample scale and timestep idxs
    idx_start = fm_solver.boundary_start_idx[global_step % args.num_boundaries]
    idx_start = torch.tensor([idx_start] * args.train_batch_size).long()

    pixel_values = batch["image"].to(device=accelerator.device)

    scales = fm_solver.scales[
        torch.argmax((idx_start[:, None] == fm_solver.boundary_start_idx).int(), dim=1)
    ]
    current_scale = scales[0].item()

    if current_scale == fm_solver.min_scale:
        previous_scales = scales
    else:
        previous_scales = fm_solver._get_previous_scale(scales)
    pixel_values = fm_solver.downscale_to_current(pixel_values, scales * 8)
    pixel_values_prev = fm_solver.downscale_to_current(
        pixel_values, previous_scales * 8
    )

    model_input_prev = vae.encode(
        pixel_values_prev.to(weight_dtype)
    ).latent_dist.sample()
    model_input_prev = (
        model_input_prev - vae.config.shift_factor
    ) * vae.config.scaling_factor
    assert model_input_prev.dtype == weight_dtype

    # Memory efficient VAE encoding
    model_inputs = []
    for pixel_value in pixel_values:
        model_input = vae.encode(
            pixel_value[None].to(weight_dtype)
        ).latent_dist.sample()
        model_input = (
            model_input - vae.config.shift_factor
        ) * vae.config.scaling_factor
        model_inputs.append(model_input)
    model_input = torch.cat(model_inputs, dim=0)
    assert model_input.dtype == weight_dtype

    batch.update(
        tokenize_captions(batch, tokenizer, tokenizer_2, tokenizer_3, is_train=True)
    )
    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        text_encoder_3=text_encoder_3,
        input_ids=batch["input_ids"],
        input_ids_2=batch["input_ids_2"],
        input_ids_3=batch["input_ids_3"],
    )

    return model_input, model_input_prev, prompt_embeds, pooled_prompt_embeds, idx_start


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def prepare_prompt_embed_from_caption(
    caption,
    tokenizer,
    tokenizer_2,
    tokenizer_3,
    text_encoder,
    text_encoder_2,
    text_encoder_3,
):
    uncond_tokens = {
        "input_ids": tokenize_prompt(tokenizer, caption),
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
class Pipeline:
    def __init__(
        self,
        vae,
        transformer,
        text_encoder,
        text_encoder_2,
        text_encoder_3,
        tokenizer,
        tokenizer_2,
        tokenizer_3,
        revision,
        variant,
        torch_dtype,
        image_processor,
    ):
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
