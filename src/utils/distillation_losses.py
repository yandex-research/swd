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

        classification_loss = F.softplus(logits_fake).mean() + F.softplus(-logits_true).mean()
    else:
        classification_loss = F.softplus(-logits_fake).mean()

    return classification_loss
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def dmd_loss(
    transformer, transformer_fake, transformer_teacher,
    prompt_embeds, pooled_prompt_embeds,
    uncond_prompt_embeds, uncond_pooled_prompt_embeds,
    model_input, timesteps_start, idx_start,
    optimizer, params_to_optimize,
    weight_dtype, noise_scheduler, fm_solver,
    accelerator, args, model_input_down=None
):
    ## STEP 1. Make the prediction with the student to create fake samples
    ## ---------------------------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    transformer.train()
    transformer_fake.eval()

    scales = fm_solver.scales[torch.argmax((idx_start[:, None] == fm_solver.boundary_start_idx).int(), dim=1)]

    if model_input_down is None:
        model_input_prev = fm_solver.downscale_to_previous_and_upscale(model_input, scales)
    else:
        model_input_prev = fm_solver.upscale_to_next(model_input_down, scales)

    noise = torch.randn_like(model_input_prev)
    noisy_model_input_curr = noise_scheduler.scale_noise(model_input_prev, timesteps_start, noise)

    with accelerator.autocast():
        model_pred = transformer(
            noisy_model_input_curr,
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps_start,
            return_dict=False,
        )[0]

    sigma_start = noise_scheduler.sigmas[idx_start].to(device=model_pred.device)[:, None, None, None]
    fake_sample = noisy_model_input_curr - sigma_start * model_pred

    ## Apply noise to the boundary points for the fake,
    idx_noisy = torch.randint(idx_start[0].item(), len(noise_scheduler.timesteps), (len(fake_sample),))
    sigma_noisy = noise_scheduler.sigmas[idx_noisy].to(device=model_pred.device)[:, None, None, None]
    timesteps_noisy = noise_scheduler.timesteps[idx_noisy].to(device=model_input.device)

    noise = torch.randn_like(fake_sample)
    noisy_fake_sample = noise_scheduler.scale_noise(fake_sample, timesteps_noisy, noise)
    ## ---------------------------------------------------------------------------

    ## STEP 2. Calculate DMD loss
    ## ---------------------------------------------------------------------------
    with torch.no_grad(), torch.autocast("cuda", dtype=weight_dtype), transformer_teacher.disable_adapter():
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
                return_dict=False
            )[0]
            real_pred = real_pred_uncond + args.cfg_teacher * (real_pred - real_pred_uncond)
        real_pred_x0 = noisy_fake_sample - sigma_noisy * real_pred

        fake_pred = transformer_fake(
            noisy_fake_sample,
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps_noisy,
            return_features=False,
            return_dict=False
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
            fake_pred = fake_pred_uncond + args.cfg_fake * (fake_pred - fake_pred_uncond)
        fake_pred_x0 = noisy_fake_sample - sigma_noisy * fake_pred

        weight_factor = abs(fake_sample.to(torch.float32) - real_pred_x0.to(torch.float32)) \
            .mean(dim=[1, 2, 3], keepdim=True).clip(min=0.00001)

    loss = (fake_pred_x0 - real_pred_x0) * noisy_fake_sample / weight_factor
    loss = torch.mean(loss)
    ## ---------------------------------------------------------------------------

    ## STEP 3. Calculate GAN loss
    ## ---------------------------------------------------------------------------
    trainable_keys = [n for n, p in transformer_fake.named_parameters() if p.requires_grad]
    transformer_fake.requires_grad_(False).eval()

    inner_features_fake = transformer_fake(
        noisy_fake_sample,
        prompt_embeds,
        pooled_prompt_embeds,
        timesteps_noisy,
        return_dict=False,
        classify_index_block=args.cls_blocks,
        return_only_features=True
    )
    gan_loss = gan_loss_fn(
        transformer_fake.module.cls_pred_branch,
        inner_features_fake,
        inner_features_true=None
    )

    loss += gan_loss * args.gen_cls_loss_weight
    ## ---------------------------------------------------------------------------

    ## STEP 4. Calculate GAN loss
    ## ---------------------------------------------------------------------------
    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()

    ## Backpropagate
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    ## ---------------------------------------------------------------------------

    transformer_fake.module.cls_pred_branch.requires_grad_(True).train()
    for n, p in transformer_fake.named_parameters():
        if n in trainable_keys:
            p.requires_grad_(True)

    return avg_loss
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def fake_diffusion_loss(
    transformer, transformer_fake,
    prompt_embeds, pooled_prompt_embeds,
    model_input, timesteps_start, idx_start,
    optimizer, params_to_optimize,
    weight_dtype, noise_scheduler, fm_solver,
    accelerator, args, model_input_down=None
):
    ## STEP 1. Make the prediction with the student to create fake samples
    ## ---------------------------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    transformer_fake.train()
    transformer.eval()

    scales = fm_solver.scales[torch.argmax((idx_start[:, None] == fm_solver.boundary_start_idx).int(), dim=1)]

    if model_input_down is None:
        model_input_prev = fm_solver.downscale_to_previous_and_upscale(model_input, scales)
    else:
        model_input_prev = fm_solver.upscale_to_next(model_input_down, scales)

    noise = torch.randn_like(model_input_prev)
    noisy_model_input_curr = noise_scheduler.scale_noise(model_input_prev, timesteps_start, noise)

    with torch.no_grad(), accelerator.autocast():
        model_pred = transformer(
            noisy_model_input_curr,
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps_start,
            return_dict=False,
        )[0]

    sigma_start = noise_scheduler.sigmas[idx_start].to(device=model_pred.device)[:, None, None, None]
    fake_sample = noisy_model_input_curr - sigma_start * model_pred

    ## Apply noise to the boundary points for the fake,
    idx_noisy = torch.randint(0, len(noise_scheduler.timesteps), (len(fake_sample),))
    timesteps_noisy = noise_scheduler.timesteps[idx_noisy].to(device=model_input.device)
    sigma_noisy = noise_scheduler.sigmas[idx_noisy].to(device=model_pred.device)[:, None, None, None]

    noise = torch.randn_like(fake_sample)
    noisy_fake_sample = noise_scheduler.scale_noise(fake_sample, timesteps_noisy, noise)
    ## ---------------------------------------------------------------------------

    ## STEP 2. Predict with fake net and calc diffusion loss
    ## ---------------------------------------------------------------------------
    with accelerator.autocast():
        fake_pred, inner_features_fake = transformer_fake(
            noisy_fake_sample,
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps_noisy,
            classify_index_block=args.cls_blocks,
            return_only_features=False,
            return_dict=False
        )
    if args.fake_diffusion_flow_pred:
        target = noise - fake_sample
        loss = F.mse_loss(fake_pred[0].float(), target.float(), reduction="mean")
    else:
        fake_pred_x0 = noisy_fake_sample - sigma_noisy * fake_pred[0]
        loss = F.mse_loss(fake_pred_x0.float(), fake_sample.float(), reduction="mean")
    ## ---------------------------------------------------------------------------

    ## STEP 3. Calculate real features and gan loss
    ## ---------------------------------------------------------------------------
    noisy_true_sample = noise_scheduler.scale_noise(model_input, timesteps_noisy, noise)
    inner_features_true = transformer_fake(
        noisy_true_sample,
        prompt_embeds,
        pooled_prompt_embeds,
        timesteps_noisy,
        classify_index_block=args.cls_blocks,
        return_only_features=True,
        return_dict=False
    )
    gan_loss = gan_loss_fn(
        transformer_fake.module.cls_pred_branch,
        inner_features_fake,
        inner_features_true
    )
    loss += gan_loss * args.guidance_cls_loss_weight
    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()

    ## Backpropagate
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    transformer.train()
    transformer_fake.eval()
    return avg_loss
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def mmd2_loss(
    x, 
    y, 
    sigma=100, 
    kernel='linear', 
    do_pdm_v2=False, 
    c=0.001,
    eps=1e-5
):
    assert x.ndim == 3
    
    if do_pdm_v2:
        # Convert [B, N, D] to [1, B*N, D]
        x = x.flatten(0, 1).unsqueeze(0)
        y = y.flatten(0, 1).unsqueeze(0)
        
    x = x.float()
    y = y.float()
    
    if kernel in ['rbf', 'energy', "laplace"]:
        
        xx = torch.bmm(x, x.transpose(1, 2))
        yy = torch.bmm(y, y.transpose(1, 2))
        xy = torch.bmm(x, y.transpose(1, 2))

        rx = torch.diagonal(xx, dim1=1, dim2=2).unsqueeze(1).expand_as(xx)
        ry = torch.diagonal(yy, dim1=1, dim2=2).unsqueeze(1).expand_as(yy)

        dxx = rx.transpose(1, 2) + rx - 2.0 * xx
        dyy = ry.transpose(1, 2) + ry - 2.0 * yy
        dxy = rx.transpose(1, 2) + ry - 2.0 * xy
            
        if kernel in ["rbf", "laplace"]:
            if kernel == "laplace":
                alpha = 1 / sigma
                dxx = dxx.sqrt().clamp(min=eps)
                dxy = dxy.sqrt().clamp(min=eps)
                dyy = dyy.sqrt().clamp(min=eps)
                          
            elif kernel == 'rbf':
                alpha = 1 / (2 * sigma**2)
                
            k_xx = torch.exp(-alpha * dxx)
            k_xy = torch.exp(-alpha * dxy)
            k_yy = torch.exp(-alpha * dyy)
            
            n = x.shape[1]
            xx_sum = (k_xx.sum(dim=(1, 2)) - n) / (n * (n - 1))
            yy_sum = (k_yy.sum(dim=(1, 2)) - n) / (n * (n - 1))
            xy_sum = k_xy.sum(dim=(1, 2)) / (n * n)
            mmd2 = xx_sum + yy_sum - 2.0 * xy_sum
            
        elif kernel == 'energy':
            
            k_xx = ((dxx + c ** 2).sqrt().clamp(min=eps) - c) # clamp is needed if c = 0
            k_xy = ((dxy + c ** 2).sqrt().clamp(min=eps) - c)
            k_yy = ((dyy + c ** 2).sqrt().clamp(min=eps) - c)
            
            xx_sum = k_xx.mean(dim=(1, 2))
            yy_sum = k_yy.mean(dim=(1, 2))
            xy_sum = k_xy.mean(dim=(1, 2))
            mmd2 = 2.0 * xy_sum - xx_sum + yy_sum
            
    elif kernel == 'linear':
        
        dxy = (x.mean(dim=1) - y.mean(dim=1)) ** 2
        mmd2 = (dxy + c ** 2).sqrt().clamp(min=eps) - c # clamp is needed if c = 0
        
    else:
        raise ValueError(f"Unsupported PDM kernel: {kernel}")
        
    return mmd2.mean()



def pdm_loss_fake(
    transformer, transformer_fake,
    prompt_embeds, pooled_prompt_embeds,
    model_input, timesteps_start, idx_start,
    optimizer, params_to_optimize,
    weight_dtype, noise_scheduler, fm_solver,
    accelerator, args, model_input_down=None
):
    ## STEP 1. Make the prediction with the student to create fake samples
    ## ---------------------------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    transformer.train()

    scales = fm_solver.scales[torch.argmax((idx_start[:, None] == fm_solver.boundary_start_idx).int(), dim=1)]

    if model_input_down is None:
        model_input_prev = fm_solver.downscale_to_previous_and_upscale(model_input, scales)
    else:
        model_input_prev = fm_solver.upscale_to_next(model_input_down, scales)

    noise = torch.randn_like(model_input_prev)
    noisy_model_input_curr = noise_scheduler.scale_noise(model_input_prev, timesteps_start, noise)

    with accelerator.autocast():
        model_pred = transformer(
            noisy_model_input_curr,
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps_start,
            return_dict=False,
        )[0]

    sigma_start = noise_scheduler.sigmas[idx_start].to(device=model_pred.device)[:, None, None, None]
    fake_sample = noisy_model_input_curr - sigma_start * model_pred
    true_sample = fm_solver.downscale_to_current(model_input, scales) if not args.do_pixels_downscale else model_input
    ## ---------------------------------------------------------------------------

    ## STEP 2. Apply noise and extract features
    ## ---------------------------------------------------------------------------
    idx_noisy = torch.randint(args.pdm_noise_start_idx, args.pdm_noise_end_idx, (len(fake_sample),))
    timesteps_noisy = noise_scheduler.timesteps[idx_noisy].to(device=model_input.device)

    noise = torch.randn_like(fake_sample)
    noisy_true_sample = noise_scheduler.scale_noise(true_sample, timesteps_noisy, noise)
    noisy_fake_sample = noise_scheduler.scale_noise(fake_sample, timesteps_noisy, noise)

    trainable_keys = [n for n, p in transformer_fake.named_parameters() if p.requires_grad]
    transformer_fake.requires_grad_(False).eval()
    
    with accelerator.autocast():
        inner_features_fake = transformer_fake(
            noisy_fake_sample,
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps_noisy,
            return_dict=False,
            classify_index_block=args.pdm_blocks
        )[0]
        inner_features_real = transformer_fake(
            noisy_true_sample,
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps_noisy,
            return_dict=False,
            classify_index_block=args.pdm_blocks
        )[0]
    ## ---------------------------------------------------------------------------

    ## STEP 3. Calculate DM loss and update the generator
    ## ---------------------------------------------------------------------------
    loss = mmd2_loss(
        inner_features_real, 
        inner_features_fake,
        sigma=args.pdm_sigma,
        do_pdm_v2=args.do_pdm_v2,
        kernel=args.pdm_kernel,
        c=args.huber_c
    )
        
    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()

    ## Backpropagate
    accelerator.backward(args.pdm_loss_weight * loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    transformer_fake.module.cls_pred_branch.requires_grad_(True).train()
    for n, p in transformer_fake.named_parameters():
        if n in trainable_keys:
            p.requires_grad_(True)
    ## ---------------------------------------------------------------------------

    return avg_loss
# ----------------------------------------------------------------------------------------------------------------------


def pdm_loss_real(
    transformer, transformer_teacher,
    prompt_embeds, pooled_prompt_embeds,
    model_input, timesteps_start, idx_start,
    optimizer, params_to_optimize,
    weight_dtype, noise_scheduler, fm_solver,
    accelerator, args, model_input_down=None
):
    ## STEP 1. Make the prediction with the student to create fake samples
    ## ---------------------------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    transformer.train()

    scales = fm_solver.scales[torch.argmax((idx_start[:, None] == fm_solver.boundary_start_idx).int(), dim=1)]

    if model_input_down is None:
        model_input_prev = fm_solver.downscale_to_previous_and_upscale(model_input, scales)
    else:
        model_input_prev = fm_solver.upscale_to_next(model_input_down, scales)

    noise = torch.randn_like(model_input_prev)
    noisy_model_input_curr = noise_scheduler.scale_noise(model_input_prev, timesteps_start, noise)

    with accelerator.autocast():
        model_pred = transformer(
            noisy_model_input_curr,
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps_start,
            return_dict=False,
        )[0]

    sigma_start = noise_scheduler.sigmas[idx_start].to(device=model_pred.device)[:, None, None, None]
    fake_sample = noisy_model_input_curr - sigma_start * model_pred
    true_sample = fm_solver.downscale_to_current(model_input, scales) if not args.do_pixels_downscale else model_input
    ## ---------------------------------------------------------------------------

    ## STEP 2. Apply noise and extract features
    ## ---------------------------------------------------------------------------
    idx_noisy = torch.randint(args.pdm_noise_start_idx, args.pdm_noise_end_idx, (len(fake_sample),))
    timesteps_noisy = noise_scheduler.timesteps[idx_noisy].to(device=model_input.device)

    noise = torch.randn_like(fake_sample)
    noisy_true_sample = noise_scheduler.scale_noise(true_sample, timesteps_noisy, noise)
    noisy_fake_sample = noise_scheduler.scale_noise(fake_sample, timesteps_noisy, noise)
    
    with accelerator.autocast():
        inner_features_fake = transformer_teacher(
            noisy_fake_sample,
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps_noisy,
            return_dict=False,
            classify_index_block=args.pdm_blocks
        )[0]
        inner_features_real = transformer_teacher(
            noisy_true_sample,
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps_noisy,
            return_dict=False,
            classify_index_block=args.pdm_blocks
        )[0]
    ## ---------------------------------------------------------------------------

    ## STEP 3. Calculate DM loss and update the generator
    ## ---------------------------------------------------------------------------
    loss = mmd2_loss(
        inner_features_real, 
        inner_features_fake,
        sigma=args.pdm_sigma,
        do_pdm_v2=args.do_pdm_v2,
        kernel=args.pdm_kernel,
        c=args.huber_c
    )
        
    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()

    ## Backpropagate
    accelerator.backward(args.pdm_loss_weight * loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    return avg_loss