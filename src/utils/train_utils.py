from typing import Optional

import torch
import numpy as np
import random
import pandas as pd
import torch.distributed as dist

from contextlib import nullcontext
from tqdm import tqdm
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from transformers import (
    CLIPTextModelWithProjection,
    T5EncoderModel,
)
from torchvision.transforms import ToPILImage

if is_wandb_available():
    import wandb


########################################################################################################################
#                                       UTILS FUNCTIONS FOR TRAIN                                                      #
########################################################################################################################


VALIDATION_PROMPTS = [
    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    'A girl with pale blue hair and a cami tank top',
    "Four cows in a pen on a sunny day",
    "Three dogs sleeping together on an unmade bed",
    "a deer with bird feathers, highly detailed, full body",
    "a masterpiece of gastronomy, a small plate, crab phalanges, cream sauce",
    "The interior of a mad scientists laboratory, Cluttered with science experiments, tools and strange machines, Eerie purple light, Close up, by Miyazaki",
    "a barred owl peeking out from dense tree branches",
    "a close-up of a blue dragonfly on a daffodil",
    "A green train is coming down the tracks",
    "A photograph of the inside of a subway train. There are frogs sitting on the seats. One of them is reading a newspaper. The window shows the river in the background.",
    "a family of four posing at the Grand Canyon",
    "A high resolution photo of a donkey in a clown costume giving a lecture at the front of a lecture hall. The blackboard has mathematical equations on it. There are many students in the lecture hall.",
    "A castle made of tortilla chips, in a river made of salsa. There are tiny burritos walking around the castle",
    "A tornado made of bees crashing into a skyscraper. painting in the style of Hokusai.",
    "A raccoon wearing formal clothes, wearing a tophat and holding a cane. The raccoon is holding a garbage bag. Oil painting in the style of abstract cubism.",
    "A castle made of cardboard.",
]


# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def log_validation(
    pipeline,
    args,
    prepare_prompt_embed_from_caption,
    solver,
    noise_scheduler,
    accelerator,
    logger,
    step,
    num_images_per_prompt=4,
):
    # Set validation prompts
    if args.validation_prompt is not None:
        logger.info(
            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
            f" {args.validation_prompt}."
        )
        validation_prompts = [args.validation_prompt]
    else:
        validation_prompts = VALIDATION_PROMPTS
        
    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    weight_dtype = pipeline.text_encoder.dtype
        
    image_logs = []
    for _, prompt in enumerate(validation_prompts):    
        # Sample batch in a loop to save memory
        prompt_embeds, pooled_prompt_embeds = prepare_prompt_embed_from_caption(
            [prompt], pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3,
            pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3,
        )
        
        sigmas = noise_scheduler.sigmas[solver.boundary_idx]
        timesteps = noise_scheduler.timesteps[solver.boundary_start_idx]
        idx_start = torch.tensor([0] * len(prompt_embeds))
        idx_end = torch.tensor([len(solver.boundary_idx) - 1] * len(prompt_embeds))
        sampling_fn = solver.flow_matching_sampling_stochastic if args.stochastic_case else solver.flow_matching_sampling
        
        images = []
        for _ in range(num_images_per_prompt):
            latent = torch.randn(
                (1, 16, 128, 128), 
                generator=generator, 
                device=accelerator.device
            )
            image = sampling_fn(
                pipeline.transformer, latent,
                prompt_embeds, pooled_prompt_embeds,
                None, None,
                idx_start, idx_end,
                cfg_scale=0.0, do_scales=True if args.scales else False,
                sigmas=sigmas, timesteps=timesteps, generator=generator
            ).to(weight_dtype)
            
            latent = (image / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
            image = pipeline.vae.decode(latent, return_dict=False)[0]
            image = pipeline.image_processor.postprocess(image, output_type='pil')[0]
            images.append(image)

        image_logs.append({"validation_prompt": prompt, "images": images})
        
    torch.cuda.empty_cache()
        
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image.resize((512, 512))))

                formatted_images = np.stack(formatted_images)
                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    return images
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def distributed_sampling(
    pipeline,
    args,
    val_prompt_path,
    prepare_prompt_embed_from_caption,
    solver,
    noise_scheduler,
    accelerator,
    logger,
    cfg_scale=0.0,
):
    logger.info(f"Running sampling")
    weight_dtype = pipeline.text_encoder.dtype
    
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    # Prepare validation prompts
    rank_batches, rank_batches_index, all_prompts = prepare_val_prompts(
        val_prompt_path, bs=args.eval_batch_size, max_cnt=args.max_eval_samples
    )

    local_images = []
    local_text_idxs = []
    
    torch.cuda.empty_cache()
                
    for cnt, mini_batch in enumerate(tqdm(rank_batches, disable=(not accelerator.is_main_process))):        
        prompt_embeds, pooled_prompt_embeds = prepare_prompt_embed_from_caption(
            list(mini_batch), pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3,
            pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3
        )

        if cfg_scale > 1.0:
            uncond_prompt_embeds, uncond_pooled_prompt_embeds = prepare_prompt_embed_from_caption(
                [' '] * len(prompt_embeds),
                pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3,
                pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3
            )
        else:
            uncond_prompt_embeds, uncond_pooled_prompt_embeds = None, None
                
        sigmas = noise_scheduler.sigmas[solver.boundary_idx]
        timesteps = noise_scheduler.timesteps[solver.boundary_start_idx]
        idx_start = torch.tensor([0] * len(prompt_embeds))
        idx_end = torch.tensor([len(solver.boundary_idx) - 1] * len(prompt_embeds))

        sampling_fn = solver.flow_matching_sampling_stochastic if args.stochastic_case else solver.flow_matching_sampling
        latent = torch.randn(
            (len(prompt_embeds), 16, 128, 128),
            generator=generator, device=accelerator.device
        )
        images = sampling_fn(
            pipeline.transformer, latent,
            prompt_embeds, pooled_prompt_embeds,
            uncond_prompt_embeds, uncond_pooled_prompt_embeds,
            idx_start, idx_end,
            cfg_scale=cfg_scale, do_scales=True if args.scales else False,
            sigmas=sigmas, timesteps=timesteps, generator=generator
        ).to(weight_dtype)

        latent = (images / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
        images = pipeline.vae.decode(latent, return_dict=False)[0]
        images = pipeline.image_processor.postprocess(images, output_type='pil')

        for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
            img_tensor = torch.tensor(np.array(images[text_idx].resize((512, 512))))
            local_images.append(img_tensor)
            local_text_idxs.append(global_idx)

    torch.cuda.empty_cache()

    local_images = torch.stack(local_images).cuda()
    local_text_idxs = torch.tensor(local_text_idxs).cuda()

    gathered_images = accelerator.gather(local_images).cpu().numpy()
    gathered_text_idxs = accelerator.gather(local_text_idxs).cpu().numpy()

    images, prompts = [], []
    if accelerator.is_main_process:
        for image, global_idx in zip(gathered_images, gathered_text_idxs):
            images.append(ToPILImage()(image))
            prompts.append(all_prompts[global_idx])

    accelerator.wait_for_everyone()
    return images, prompts
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def prepare_val_prompts(path, bs=20, max_cnt=5000):
    df = pd.read_csv(path)
    all_text = list(df['caption'])
    all_text = all_text[:max_cnt]

    num_batches = ((len(all_text) - 1) // (bs * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = np.array_split(np.array(all_text), num_batches)
    rank_batches = all_batches[dist.get_rank():: dist.get_world_size()]

    index_list = np.arange(len(all_text))
    all_batches_index = np.array_split(index_list, num_batches)
    rank_batches_index = all_batches_index[dist.get_rank():: dist.get_world_size()]
    return rank_batches, rank_batches_index, all_text
# ----------------------------------------------------------------------------------------------------------------------


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
    # In case there are precomputed text embeds - use them
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_embeds_2: Optional[torch.Tensor] = None,
    t5_prompt_embeds: Optional[torch.Tensor] = None,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    pooled_prompt_embeds_2: Optional[torch.Tensor] = None,
    device='cuda'
):
    # Prepare CLIP prompt embeds
    if prompt_embeds is None:
        prompt_embeds = text_encoder(input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
    else:
        prompt_embeds, pooled_prompt_embeds = prompt_embeds.to(device), pooled_prompt_embeds.to(device)
    if prompt_embeds_2 is None:
        prompt_embeds_2 = text_encoder_2(input_ids_2.to(device), output_hidden_states=True)
        pooled_prompt_embeds_2 = prompt_embeds_2[0]
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]
    else:
        prompt_embeds_2, pooled_prompt_embeds_2 = prompt_embeds_2.to(device), pooled_prompt_embeds_2.to(device)
    # Prepare T5 prompt embeds
    if t5_prompt_embeds is None:
        t5_prompt_embeds = text_encoder_3(input_ids_3.to(device))[0]
    else:
        t5_prompt_embeds = t5_prompt_embeds.to(device)
    # Join prompt embeds
    clip_prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embeds.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embeds], dim=-2)
    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, pooled_prompt_embeds_2], dim=-1)
    return prompt_embeds, pooled_prompt_embeds
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def tokenize_captions(examples, args, tokenizer, tokenizer_2, tokenizer_3, is_train=True):
    captions = []
    for caption in examples[args.text_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{args.text_column}` should contain either strings or lists of strings."
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
def get_logit_normal_probs(num_sizes: int, loc: float, scale: float, eps: float = 1e-5) -> torch.Tensor:
    ts = torch.linspace(0, 1, steps=num_sizes)
    logit_ts = ((ts + eps) / (1 - ts + eps)).log()
    numerator = (logit_ts - loc).div(2 * scale**2).mul_(-1).exp()
    denominator = 1 / (scale * (ts + eps) * (1 - ts + eps))
    probs = numerator / denominator
    return probs
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model
# ----------------------------------------------------------------------------------------------------------------------
