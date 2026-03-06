import os

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torchvision.transforms import ToPILImage
from tqdm import tqdm

VALIDATION_PROMPTS = [
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "A girl with pale blue hair and a cami tank top",
    "Four cows in a pen on a sunny day",
    "Three dogs sleeping together on an unmade bed",
    "a deer with bird feathers, highly detailed, full body",
    "The interior of a mad scientists laboratory, Cluttered with science experiments, tools and strange machines, Eerie purple light, Close up, by Miyazaki",
    "a barred owl peeking out from dense tree branches",
    "a close-up of a blue dragonfly on a daffodil",
    "A green train is coming down the tracks",
    "A photograph of the inside of a subway train. There are frogs sitting on the seats. One of them is reading a newspaper. The window shows the river in the background.",
    "a family of four posing at the Grand Canyon",
    "A high resolution photo of a donkey in a clown costume giving a lecture at the front of a lecture hall. The blackboard has mathematical equations on it. There are many students in the lecture hall.",
    "A castle made of tortilla chips, in a river made of salsa. There are tiny burritos walking around the castle",
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
            f"Running validation... \n Generating {args.num_validation_images} images with prompt: "
            f" {args.validation_prompt}."
        )
        validation_prompts = [args.validation_prompt]
    else:
        validation_prompts = VALIDATION_PROMPTS

    # run inference
    generator = (
        torch.Generator(device=accelerator.device).manual_seed(args.seed)
        if args.seed
        else None
    )
    weight_dtype = pipeline.text_encoder.dtype

    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images", str(step)), exist_ok=True)

    for prompt in validation_prompts:
        os.makedirs(
            os.path.join(args.output_dir, "images", str(step), prompt[:30]),
            exist_ok=True,
        )
        # Sample batch in a loop to save memory
        prompt_embeds, pooled_prompt_embeds = prepare_prompt_embed_from_caption(
            [prompt],
            pipeline.tokenizer,
            pipeline.tokenizer_2,
            pipeline.tokenizer_3,
            pipeline.text_encoder,
            pipeline.text_encoder_2,
            pipeline.text_encoder_3,
        )

        sigmas = noise_scheduler.sigmas[solver.boundary_idx]
        timesteps = noise_scheduler.timesteps[solver.boundary_start_idx]
        idx_start = torch.tensor([0] * len(prompt_embeds))
        idx_end = torch.tensor([len(solver.boundary_idx) - 1] * len(prompt_embeds))

        for i in range(num_images_per_prompt):
            latent = torch.randn(
                (1, 16, 128, 128), generator=generator, device=accelerator.device
            )
            image = solver.sampling(
                pipeline.transformer,
                latent,
                prompt_embeds,
                pooled_prompt_embeds,
                None,
                None,
                idx_start,
                idx_end,
                cfg_scale=0.0,
                do_scales=True if args.scales else False,
                sigmas=sigmas,
                timesteps=timesteps,
                generator=generator,
            ).to(weight_dtype)

            latent = (
                image / pipeline.vae.config.scaling_factor
            ) + pipeline.vae.config.shift_factor
            image = pipeline.vae.decode(latent, return_dict=False)[0]
            image = pipeline.image_processor.postprocess(image, output_type="pil")[0]
            image.save(
                os.path.join(
                    args.output_dir, "images", str(step), prompt[:30], f"{i}.png"
                )
            )


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
    logger.info("Running sampling")
    weight_dtype = pipeline.text_encoder.dtype

    generator = (
        torch.Generator(device=accelerator.device).manual_seed(args.seed)
        if args.seed
        else None
    )
    # Prepare validation prompts
    rank_batches, rank_batches_index, all_prompts = prepare_val_prompts(
        val_prompt_path, bs=args.eval_batch_size, max_cnt=args.max_eval_samples
    )

    local_images = []
    local_text_idxs = []

    torch.cuda.empty_cache()

    for cnt, mini_batch in enumerate(
        tqdm(rank_batches, disable=(not accelerator.is_main_process))
    ):
        prompt_embeds, pooled_prompt_embeds = prepare_prompt_embed_from_caption(
            list(mini_batch),
            pipeline.tokenizer,
            pipeline.tokenizer_2,
            pipeline.tokenizer_3,
            pipeline.text_encoder,
            pipeline.text_encoder_2,
            pipeline.text_encoder_3,
        )

        if cfg_scale > 1.0:
            (
                uncond_prompt_embeds,
                uncond_pooled_prompt_embeds,
            ) = prepare_prompt_embed_from_caption(
                [" "] * len(prompt_embeds),
                pipeline.tokenizer,
                pipeline.tokenizer_2,
                pipeline.tokenizer_3,
                pipeline.text_encoder,
                pipeline.text_encoder_2,
                pipeline.text_encoder_3,
            )
        else:
            uncond_prompt_embeds, uncond_pooled_prompt_embeds = None, None

        sigmas = noise_scheduler.sigmas[solver.boundary_idx]
        timesteps = noise_scheduler.timesteps[solver.boundary_start_idx]
        idx_start = torch.tensor([0] * len(prompt_embeds))
        idx_end = torch.tensor([len(solver.boundary_idx) - 1] * len(prompt_embeds))

        latent = torch.randn(
            (len(prompt_embeds), 16, 128, 128),
            generator=generator,
            device=accelerator.device,
        )
        images = solver.sampling(
            pipeline.transformer,
            latent,
            prompt_embeds,
            pooled_prompt_embeds,
            uncond_prompt_embeds,
            uncond_pooled_prompt_embeds,
            idx_start,
            idx_end,
            cfg_scale=cfg_scale,
            do_scales=True if args.scales else False,
            sigmas=sigmas,
            timesteps=timesteps,
            generator=generator,
        ).to(weight_dtype)

        latent = (
            images / pipeline.vae.config.scaling_factor
        ) + pipeline.vae.config.shift_factor
        images = pipeline.vae.decode(latent, return_dict=False)[0]
        images = pipeline.image_processor.postprocess(images, output_type="pil")

        for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
            img_tensor = torch.tensor(np.array(images[text_idx].resize((512, 512))))
            local_images.append(img_tensor)
            local_text_idxs.append(global_idx)

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
    all_text = list(df["caption"])
    all_text = all_text[:max_cnt]

    num_batches = (
        (len(all_text) - 1) // (bs * dist.get_world_size()) + 1
    ) * dist.get_world_size()
    all_batches = np.array_split(np.array(all_text), num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    index_list = np.arange(len(all_text))
    all_batches_index = np.array_split(index_list, num_batches)
    rank_batches_index = all_batches_index[dist.get_rank() :: dist.get_world_size()]
    return rank_batches, rank_batches_index, all_text


# ----------------------------------------------------------------------------------------------------------------------
