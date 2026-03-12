import types
from pathlib import Path

import torch
from accelerate.logging import get_logger
from diffusers.image_processor import VaeImageProcessor
from tqdm.auto import tqdm

from src.dataset import get_loader
from src.losses import fake_diffusion_loss, generator_loss
from src.evaluation.eval import distributed_sampling, log_validation
from src.evaluation.metrics import calculate_scores
from src.flow_matching_sampler import FlowMatchingSolver
from src.transformer_with_discriminator import TransformerCls, forward_with_feature_extraction
from src.utils.prepare_utils import (
    prepare_accelerator,
    prepare_models,
    prepare_optimizer,
)
from src.utils.setup_utils import (
    load_if_exist,
    prepare_3rd_party,
    saving,
    seed_everything,
    set_tf32,
)
from src.utils.train_utils import (
    Pipeline,
    prepare_prompt_embed_from_caption,
    sample_batch,
    unwrap_model,
)

logger = get_logger(__name__)


########################################################################################################################
#                                               TRAINING                                                               #
########################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
def train(args):
    ## PREPARATION STAGE
    ## -----------------------------------------------------------------------------------------------
    ## Prepare accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = prepare_accelerator(args, logging_dir, False)

    # Some useful asserts
    assert (
        args.max_eval_samples % accelerator.num_processes == 0
    ), "Must be divisible by world size. Otherwise, allgather fails."

    # Fix seed and setup tf32
    seed_everything(args.seed, accelerator.process_index, accelerator.num_processes)
    set_tf32(tf32=True)

    ## Prepare models
    (
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
    ) = prepare_models(args, accelerator)

    ## Set up schedulers: diffusion and distilled models
    fm_solver = FlowMatchingSolver(
        noise_scheduler,
        args.num_timesteps,
        args.num_boundaries,
        args.scales,
        args.boundaries,
    )

    ## Add GAN head
    transformer_fake.forward = types.MethodType(forward_with_feature_extraction, transformer_fake)

    transformer_fake = TransformerCls(args, transformer_fake)
    initial_global_step = load_if_exist(args, accelerator, transformer, is_student=True)
    _ = load_if_exist(args, accelerator, transformer_fake, is_student=False)
    transformer, transformer_fake = accelerator.prepare(transformer, transformer_fake)

    ## Prepare optimizers
    optimizer, params_to_optimize = prepare_optimizer(
        args, transformer, is_student=True
    )

    optimizer_fake, params_to_optimize_fake = prepare_optimizer(
        args, transformer_fake, is_student=False
    )
    optimizer, optimizer_fake = accelerator.prepare(optimizer, optimizer_fake)

    ## Prepare data
    if args.model_name == "medium":
        root_dir = "data/sd35_medium_train_data"
    elif args.model_name == "large":
        root_dir = "data/sd35_large_train_data"
    train_dataloader, train_dataset = get_loader(
        args.train_batch_size, root_dir=root_dir
    )

    ## Prepare 3rd party utils
    prepare_3rd_party(args, accelerator)
    ## -----------------------------------------------------------------------------------------------

    ## TRAINING STAGE
    ## -----------------------------------------------------------------------------------------------
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    if hasattr(train_dataset, "__len__"):
        logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = initial_global_step
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    (
        uncond_prompt_embeds,
        uncond_pooled_prompt_embeds,
    ) = prepare_prompt_embed_from_caption(
        [""] * args.train_batch_size,
        tokenizer,
        tokenizer_2,
        tokenizer_3,
        text_encoder,
        text_encoder_2,
        text_encoder_3,
    )

    assert transformer.training
    while global_step < args.max_train_steps:
        (
            model_input,
            model_input_prev,
            prompt_embeds,
            pooled_prompt_embeds,
            idx_start,
        ) = sample_batch(
            args,
            accelerator,
            global_step,
            train_dataloader,
            fm_solver,
            tokenizer,
            tokenizer_2,
            tokenizer_3,
            text_encoder,
            text_encoder_2,
            text_encoder_3,
            vae,
            weight_dtype,
        )
        timesteps_start = noise_scheduler.timesteps[idx_start].to(
            device=model_input.device
        )

        ### DMD loss
        ### ----------------------------------------------------
        if args.do_dmd_loss:
            for _ in range(args.n_steps_fake_dmd):
                avg_dmd_fake_loss = fake_diffusion_loss(
                    transformer,
                    transformer_fake,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    model_input,
                    timesteps_start,
                    idx_start,
                    optimizer_fake,
                    params_to_optimize_fake,
                    weight_dtype,
                    noise_scheduler,
                    fm_solver,
                    accelerator,
                    args,
                    model_input_down=model_input_prev,
                )

                (
                    model_input,
                    model_input_prev,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    idx_start,
                ) = sample_batch(
                    args,
                    accelerator,
                    global_step,
                    train_dataloader,
                    fm_solver,
                    tokenizer,
                    tokenizer_2,
                    tokenizer_3,
                    text_encoder,
                    text_encoder_2,
                    text_encoder_3,
                    vae,
                    weight_dtype,
                )
                timesteps_start = noise_scheduler.timesteps[idx_start].to(
                    device=model_input.device
                )

        avg_dmd_loss, avg_mmd_loss = generator_loss(
            transformer,
            transformer_fake,
            transformer_teacher,
            prompt_embeds,
            pooled_prompt_embeds,
            uncond_prompt_embeds,
            uncond_pooled_prompt_embeds,
            model_input,
            timesteps_start,
            idx_start,
            optimizer,
            params_to_optimize,
            weight_dtype,
            noise_scheduler,
            fm_solver,
            accelerator,
            args,
            model_input_down=model_input_prev,
        )
        ### ----------------------------------------------------

        progress_bar.update(1)
        global_step += 1

        ### Model evaluation
        ### ----------------------------------------------------
        if global_step % args.evaluation_steps == 0:
            image_processor = VaeImageProcessor(
                vae_scale_factor=vae.config.scaling_factor
            )
            pipeline = Pipeline(
                vae=vae,
                transformer=unwrap_model(transformer, accelerator),
                text_encoder=unwrap_model(text_encoder, accelerator),
                text_encoder_2=unwrap_model(text_encoder_2, accelerator),
                text_encoder_3=unwrap_model(text_encoder_3, accelerator),
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                tokenizer_3=tokenizer_3,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
                image_processor=image_processor,
            )

            for eval_set_name in ["mjhq", "coco"]:
                eval_prompts_path = f"src/prompts/{eval_set_name}.csv"
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
                    pick_score, clip_score, fid_score = calculate_scores(
                        args,
                        images,
                        prompts,
                        ref_stats_path=fid_stats_path,
                    )
                    logs = {
                        f"fid_{eval_set_name}": fid_score.item(),
                        f"pick_score_{eval_set_name}": pick_score.item(),
                        f"clip_score_{eval_set_name}": clip_score.item(),
                    }
                    print(eval_set_name, logs)
                    accelerator.log(logs, step=global_step)

                torch.cuda.empty_cache()
                accelerator.wait_for_everyone()
        ### ----------------------------------------------------

        ### Saving checkpoint
        if accelerator.is_main_process and global_step % args.evaluation_steps == 0:
            saving(transformer, args, accelerator, global_step, is_student=True)
            saving(transformer_fake, args, accelerator, global_step, is_student=False)
        accelerator.wait_for_everyone()

        ### Log validation images
        ### ----------------------------------------------------
        if accelerator.is_main_process and global_step % args.validation_steps == 0:
            image_processor = VaeImageProcessor(
                vae_scale_factor=vae.config.scaling_factor
            )
            pipeline = Pipeline(
                vae=vae,
                transformer=unwrap_model(transformer, accelerator),
                text_encoder=unwrap_model(text_encoder, accelerator),
                text_encoder_2=unwrap_model(text_encoder_2, accelerator),
                text_encoder_3=unwrap_model(text_encoder_3, accelerator),
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                tokenizer_3=tokenizer_3,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
                image_processor=image_processor,
            )

            log_validation(
                pipeline,
                args,
                prepare_prompt_embed_from_caption,
                fm_solver,
                noise_scheduler,
                accelerator,
                logger,
                global_step,
            )

            del pipeline
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        ### ----------------------------------------------------
        if accelerator.is_main_process and global_step % args.log_steps == 0:
            logs = {
                "fake_loss": avg_dmd_fake_loss.detach().item()
                if args.do_dmd_loss
                else 0,
                "dmd_loss": avg_dmd_loss.detach().item() if args.do_dmd_loss else 0,
                "mmd_loss": avg_mmd_loss.detach().item() if args.do_mmd_loss else 0,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

    ## -----------------------------------------------------------------------------------------------

    accelerator.wait_for_everyone()
    accelerator.end_training()


# ----------------------------------------------------------------------------------------------------------------------
