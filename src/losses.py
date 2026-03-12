import torch
import torch.nn.functional as F

########################################################################################################################
#                            THE LOSSES NEEDED FOR THE DISTILLATION OF SD3                                            #
########################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
def gan_loss_fn(cls_head, inner_features_fake, inner_features_true=None):
    logits_fake = 0
    for x in inner_features_fake:
        logits_fake += cls_head(x.float().mean(dim=1))
    logits_fake /= len(inner_features_fake)

    if inner_features_true is not None:
        logits_true = 0
        for x in inner_features_true:
            logits_true += cls_head(x.float().mean(dim=1))
        logits_true /= len(inner_features_true)

        classification_loss = (
            F.softplus(logits_fake).mean() + F.softplus(-logits_true).mean()
        )
    else:
        classification_loss = F.softplus(-logits_fake).mean()

    return classification_loss


# ---------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def fake_diffusion_loss(
    transformer,
    transformer_fake,
    prompt_embeds,
    pooled_prompt_embeds,
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
    model_input_down=None,
):
    ## STEP 1. Make the prediction with the student to create fake samples
    ## ---------------------------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    transformer_fake.train()
    transformer.eval()

    scales = fm_solver.scales[
        torch.argmax((idx_start[:, None] == fm_solver.boundary_start_idx).int(), dim=1)
    ]

    if model_input_down is None:
        model_input_prev = fm_solver.downscale_to_previous_and_upscale(
            model_input, scales
        )
    else:
        model_input_prev = fm_solver.upscale_to_next(model_input_down, scales)

    noise = torch.randn_like(model_input_prev)
    noisy_model_input_curr = noise_scheduler.scale_noise(
        model_input_prev, timesteps_start, noise
    )

    with torch.no_grad(), accelerator.autocast():
        model_pred = transformer(
            noisy_model_input_curr,
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps_start,
            return_dict=False,
        )[0]

    sigma_start = noise_scheduler.sigmas[idx_start].to(device=model_pred.device)[
        :, None, None, None
    ]
    fake_sample = noisy_model_input_curr - sigma_start * model_pred

    ## Apply noise to the boundary points for the fake
    idx_noisy = torch.randint(0, len(noise_scheduler.timesteps), (len(fake_sample),))
    timesteps_noisy = noise_scheduler.timesteps[idx_noisy].to(device=model_input.device)
    sigma_noisy = noise_scheduler.sigmas[idx_noisy].to(device=model_pred.device)[
        :, None, None, None
    ]

    noise = torch.randn_like(fake_sample)
    noisy_fake_sample = noise_scheduler.scale_noise(fake_sample, timesteps_noisy, noise)
    ## ---------------------------------------------------------------------------

    ## STEP 2. Predict with fake net and calculate diffusion loss
    ## ---------------------------------------------------------------------------
    with accelerator.autocast():
        fake_pred, inner_features_fake = transformer_fake(
            noisy_fake_sample,
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps_noisy,
            classify_index_block=args.cls_blocks,
            return_only_features=False,
            return_dict=False,
        )

    fake_pred_x0 = noisy_fake_sample - sigma_noisy * fake_pred[0]
    loss = F.mse_loss(fake_pred_x0.float(), fake_sample.float(), reduction="mean")
    ## ---------------------------------------------------------------------------

    ## STEP 3. Calculate real features and gan loss
    ## ---------------------------------------------------------------------------
    noisy_true_sample = noise_scheduler.scale_noise(model_input, timesteps_noisy, noise)
    with accelerator.autocast():
        inner_features_true = transformer_fake(
            noisy_true_sample,
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps_noisy,
            classify_index_block=args.cls_blocks,
            return_only_features=True,
            return_dict=False,
        )
    gan_loss = gan_loss_fn(
        transformer_fake.module.cls_pred_branch,
        inner_features_fake,
        inner_features_true,
    )
    loss += gan_loss * args.disc_cls_loss_weight
    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()

    ## Backpropagate
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
    optimizer.step()

    transformer.train()
    transformer_fake.eval()
    return avg_loss


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def generator_loss(
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
    model_input_down=None,
):
    ## STEP 1. Make the prediction with the student to create fake samples
    ## ---------------------------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    transformer.train()
    transformer_fake.eval()

    scales = fm_solver.scales[
        torch.argmax((idx_start[:, None] == fm_solver.boundary_start_idx).int(), dim=1)
    ]

    if model_input_down is None:
        model_input_prev = fm_solver.downscale_to_previous_and_upscale(
            model_input, scales
        )
    else:
        model_input_prev = fm_solver.upscale_to_next(model_input_down, scales)

    noise = torch.randn_like(model_input_prev)
    noisy_model_input_curr = noise_scheduler.scale_noise(
        model_input_prev, timesteps_start, noise
    )

    with accelerator.autocast():
        model_pred = transformer(
            noisy_model_input_curr,
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps_start,
            return_dict=False,
        )[0]

    sigma_start = noise_scheduler.sigmas[idx_start].to(device=model_pred.device)[
        :, None, None, None
    ]
    fake_sample = noisy_model_input_curr - sigma_start * model_pred
    true_sample = model_input

    if args.do_dmd_loss:
        ## Apply noise to the boundary points for the fake,
        idx_noisy = torch.randint(
            args.dmd_noise_start_idx, args.dmd_noise_end_idx, (len(fake_sample),)
        )
        sigma_noisy = noise_scheduler.sigmas[idx_noisy].to(device=model_pred.device)[
            :, None, None, None
        ]
        timesteps_noisy = noise_scheduler.timesteps[idx_noisy].to(
            device=model_input.device
        )

        noise = torch.randn_like(fake_sample)
        noisy_fake_sample = noise_scheduler.scale_noise(
            fake_sample, timesteps_noisy, noise
        )
        ## ---------------------------------------------------------------------------

        ## STEP 2. Calculate DMD loss
        ## ---------------------------------------------------------------------------
        with torch.no_grad(), accelerator.autocast(), transformer_teacher.disable_adapter():
            real_pred = transformer_teacher(
                noisy_fake_sample,
                prompt_embeds,
                pooled_prompt_embeds,
                timesteps_noisy,
                return_dict=False,
            )[0]

            if args.cfg_teacher > 1.0:
                real_pred_uncond = transformer_teacher(
                    noisy_fake_sample,
                    uncond_prompt_embeds,
                    uncond_pooled_prompt_embeds,
                    timesteps_noisy,
                    return_dict=False,
                )[0]
                real_pred = real_pred_uncond + args.cfg_teacher * (
                    real_pred - real_pred_uncond
                )
            real_pred_x0 = noisy_fake_sample - sigma_noisy * real_pred

            fake_pred = transformer_fake(
                noisy_fake_sample,
                prompt_embeds,
                pooled_prompt_embeds,
                timesteps_noisy,
                return_features=False,
                return_dict=False,
            )[0]

            if args.cfg_fake > 1.0:
                fake_pred_uncond = transformer_fake(
                    noisy_fake_sample,
                    uncond_prompt_embeds,
                    uncond_pooled_prompt_embeds,
                    timesteps_noisy,
                    return_features=False,
                    return_dict=False,
                )[0]
                fake_pred = fake_pred_uncond + args.cfg_fake * (
                    fake_pred - fake_pred_uncond
                )

            fake_pred_x0 = noisy_fake_sample - sigma_noisy * fake_pred

        p_real = fake_sample - real_pred_x0
        p_fake = fake_sample - fake_pred_x0

        grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True)
        grad = torch.nan_to_num(grad)

        loss = 0.5 * F.mse_loss(
            fake_sample.float(), (fake_sample - grad).detach().float(), reduction="mean"
        )
        loss = args.dmd_loss_weight * loss
    else:
        loss = torch.zeros(1, device=fake_sample.device, dtype=fake_sample.dtype)

    ## ---------------------------------------------------------------------------

    ## STEP 3. Calculate GAN loss
    ## ---------------------------------------------------------------------------
    trainable_keys = [
        n for n, p in transformer_fake.named_parameters() if p.requires_grad
    ]
    transformer_fake.requires_grad_(False).eval()

    if args.do_gan_loss:
        inner_features_fake = transformer_fake(
            noisy_fake_sample,
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps_noisy,
            return_dict=False,
            classify_index_block=args.cls_blocks,
            return_only_features=True,
        )
        gan_loss = gan_loss_fn(
            transformer_fake.module.cls_pred_branch,
            inner_features_fake,
            inner_features_true=None,
        )

        loss += gan_loss * args.gen_cls_loss_weight

    ## ---------------------------------------------------------------------------

    ## STEP 4. Calculate MMD loss
    ## ---------------------------------------------------------------------------
    if args.do_mmd_loss:
        idx_noisy = torch.randint(
            args.mmd_noise_start_idx, args.mmd_noise_end_idx, (len(fake_sample),)
        )
        timesteps_noisy = noise_scheduler.timesteps[idx_noisy].to(
            device=model_input.device
        )

        noise = torch.randn_like(fake_sample)
        noisy_true_sample = noise_scheduler.scale_noise(
            true_sample, timesteps_noisy, noise
        )
        noisy_fake_sample = noise_scheduler.scale_noise(
            fake_sample, timesteps_noisy, noise
        )

        with accelerator.autocast():
            inner_features_fake = transformer_fake(
                noisy_fake_sample,
                prompt_embeds,
                pooled_prompt_embeds,
                timesteps_noisy,
                return_dict=False,
                classify_index_block=args.mmd_blocks,
            )[0]
            inner_features_real = transformer_fake(
                noisy_true_sample,
                prompt_embeds,
                pooled_prompt_embeds,
                timesteps_noisy,
                return_dict=False,
                classify_index_block=args.mmd_blocks,
            )[0]
        mmd_loss = mmd_loss_(
            inner_features_real,
            inner_features_fake,
            kernel=args.mmd_kernel,
            sigma=args.mmd_rbf_sigma,
            do_batch_mmd=args.do_batch_mmd,
            c=args.huber_c,
        )
        loss += args.mmd_loss_weight * mmd_loss
    else:
        mmd_loss = torch.zeros_like(loss)

    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
    avg_mmd_loss = accelerator.gather(mmd_loss.repeat(args.train_batch_size)).mean()

    ## Backpropagate
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
    optimizer.step()
    ## ---------------------------------------------------------------------------

    transformer_fake.module.cls_pred_branch.requires_grad_(True).train()
    for n, p in transformer_fake.named_parameters():
        if n in trainable_keys:
            p.requires_grad_(True)

    return avg_loss, avg_mmd_loss


# ---------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def mmd_loss_(x, y, kernel="linear", sigma=100, do_batch_mmd=False, c=0.001, eps=1e-5):
    assert x.ndim == 3

    if do_batch_mmd:
        # Convert [B, N, D] to [1, B*N, D]
        x = x.flatten(0, 1).unsqueeze(0)
        y = y.flatten(0, 1).unsqueeze(0)

    x = x.float()
    y = y.float()

    if kernel == "rbf":
        xx = torch.bmm(x, x.transpose(1, 2))
        yy = torch.bmm(y, y.transpose(1, 2))
        xy = torch.bmm(x, y.transpose(1, 2))

        rx = torch.diagonal(xx, dim1=1, dim2=2).unsqueeze(1).expand_as(xx)
        ry = torch.diagonal(yy, dim1=1, dim2=2).unsqueeze(1).expand_as(yy)

        dxx = rx.transpose(1, 2) + rx - 2.0 * xx
        dyy = ry.transpose(1, 2) + ry - 2.0 * yy
        dxy = rx.transpose(1, 2) + ry - 2.0 * xy

        alpha = 1 / (2 * sigma**2)

        k_xx = torch.exp(-alpha * dxx)
        k_xy = torch.exp(-alpha * dxy)
        k_yy = torch.exp(-alpha * dyy)

        n = x.shape[1]
        xx_sum = (k_xx.sum(dim=(1, 2)) - n) / (n * (n - 1))
        yy_sum = (k_yy.sum(dim=(1, 2)) - n) / (n * (n - 1))
        xy_sum = k_xy.sum(dim=(1, 2)) / (n * n)
        mmd = xx_sum + yy_sum - 2.0 * xy_sum

    elif kernel == "linear":
        dxy = (x.mean(dim=1) - y.mean(dim=1)) ** 2
        mmd = (dxy + c**2).sqrt().clamp(min=eps) - c  # clamp is needed if c = 0

    else:
        raise ValueError(f"Unsupported MMD kernel: {kernel}")

    return mmd.mean()
