import torch

########################################################################################################################
#                                            SAMPLER UTILS                                                             #
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
class FlowMatchingSolver:

    ## ---------------------------------------------------------------------------
    def __init__(
        self,
        noise_scheduler,
        num_boundaries=1,
        scales=None,
        boundaries=None
    ):
        """
        Set up boundaries indexes.
        For example, if num_boundaries = 3
        boundary_start_idx = [ 0,  9, 18], boundary_end_idx = [9, 18, 28]

        :param noise_scheduler: scheduler from the diffusers
        :param num_boundaries: (int)
        """
        noise_scheduler.set_timesteps(28)  ## HARDCODED AS A COMMON VALUE
        
        if scales:
            assert len(scales) == num_boundaries
            self.min_scale = scales[0]
            scales = torch.tensor(scales)
            self.scales_pixels = scales * 8
        self.scales = scales

        self.num_boundaries = num_boundaries
        if num_boundaries == 0:
            boundary_idx = torch.tensor([0])
            self.boundary_start_idx = boundary_idx
        else:
            if boundaries is None:
                self.boundary_idx = torch.linspace(0,
                                                   len(noise_scheduler.timesteps),
                                                   num_boundaries + 1, dtype=int)
            else:
                self.boundary_idx = torch.tensor(boundaries, dtype=int)
            self.boundary_start_idx = self.boundary_idx[:-1]
            self.boundary_end_idx = self.boundary_idx[1:]

        self.noise_scheduler = noise_scheduler
    ## ---------------------------------------------------------------------------


    ## ---------------------------------------------------------------------------
    def sample_end_boundary_idx(self, batch_of_start_idx):
        """
        Sample indexes of the end boundaries for batch of start indexes.
        For example, if num_boundaries = 3 and batch_of_start_idx = [18,  0,  0,  9].
        Then, batch_of_end_idx = [28,  9,  9, 18]

        :param batch_of_start_idx: (tensor), [b_size]
        :return: batch_of_end_idx (tensor), [b_size]
        """

        mask = (batch_of_start_idx[None, :] == self.boundary_end_idx[:, None]).long()
        idx = torch.argmax(mask[[self.num_boundaries - 1] + list(range(0, self.num_boundaries - 1)), :], dim=0)
        batch_of_end_idx = self.boundary_end_idx[idx]

        return batch_of_end_idx
    ## ---------------------------------------------------------------------------


    ## ---------------------------------------------------------------------------
    def _get_previous_scale(self, batch_of_scales):
        scales_idx = torch.argmax((batch_of_scales[:, None] == self.scales).int(), dim=1)
        previous_scales_idx = scales_idx - 1
        previous_scales = self.scales[previous_scales_idx]
        return previous_scales
    ## ---------------------------------------------------------------------------


    ## ---------------------------------------------------------------------------
    def downscale_to_previous_and_upscale(self, sample, scales):
        """
        We downscale the sample to the previous scales and upscale to the current scales.
        :param scales: torch.tensor(), [b_size], batch of scales on the current interval
        """
        if scales[0].item() == self.min_scale:
            sample = torch.nn.functional.interpolate(sample, size=int(scales[0].item()), mode='area')
            return sample

        previous_scales = self._get_previous_scale(scales)
        sample = torch.nn.functional.interpolate(sample, size=int(previous_scales[0].item()), mode='area')
        sample = torch.nn.functional.interpolate(sample, size=int(scales[0].item()), mode='bicubic')
        return sample
    ## ---------------------------------------------------------------------------


    ## ---------------------------------------------------------------------------
    def upscale_to_next(self, sample, scales):
        """
        We upscale the sample to the next scales
        :param scales: torch.tensor(), [b_size], batch of scales on the current interval
        """
        if scales[0].item() == self.min_scale:
            return sample

        sample = torch.nn.functional.interpolate(sample, size=int(scales[0].item()), mode='bicubic')        
        return sample
    ## ---------------------------------------------------------------------------


    ## ---------------------------------------------------------------------------
    def downscale_to_current(self, sample, scales):
        """
        We downscale the sample to the current scales
        :param scales: torch.tensor(), [b_size], batch of scales on the current interval
        """
        sample = torch.nn.functional.interpolate(sample, size=int(scales[0].item()), mode='area')
        return sample
    ## ---------------------------------------------------------------------------


    ## ---------------------------------------------------------------------------
    def flow_matching_single_step(self, sample, model_output, sigma, sigma_next):
        prev_sample = sample + (sigma_next - sigma) * model_output
        return prev_sample
    ## ---------------------------------------------------------------------------


    ## ---------------------------------------------------------------------------
    @torch.no_grad()
    def flow_matching_sampling_stochastic(
        self, model, latent,                                  
        prompt_embeds, pooled_prompt_embeds,
        uncond_prompt_embeds, uncond_pooled_prompt_embeds,
        idx_start, idx_end,
        cfg_scale=7.0, do_scales=True,
        sigmas=None, timesteps=None, generator=None
    ):
        weight_dtype = prompt_embeds.dtype
        sigmas = self.noise_scheduler.sigmas if sigmas is None else sigmas
        timesteps = self.noise_scheduler.timesteps if timesteps is None else timesteps
        if do_scales:
            latent = torch.randn((len(prompt_embeds), 16, int(self.min_scale), int(self.min_scale)),
                                 generator=generator, device=latent.device)
            try:
                next_scale = self.scales[1]
            except IndexError:
                next_scale = 128
            k = 1

        while True:
            timestep = timesteps[idx_start].to(device=latent.device)
            sigma = sigmas[idx_start].to(device=latent.device)
            sigma_next = sigmas[idx_start + 1].to(device=latent.device)

            with torch.autocast("cuda", dtype=weight_dtype):
                noise_pred = model(
                        latent,
                        prompt_embeds,
                        pooled_prompt_embeds,
                        timestep,
                        return_dict=False,
                )[0]
                if cfg_scale > 1.0:
                    noise_pred_uncond = model(
                            latent,
                            uncond_prompt_embeds,
                            uncond_pooled_prompt_embeds,
                            timestep,
                            return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)

            latent = latent - noise_pred * sigma[:, None, None, None]  # 32, 48, 64

            if (idx_start + 1)[0].item() == idx_end[0].item():
                break
            idx_start = idx_start + 1

            if do_scales:
                if int(self.scales[k - 1]) != int(next_scale):
                    latent = torch.nn.functional.interpolate(latent, size=int(next_scale), mode='bicubic') 
                
                noise = torch.randn(*latent.shape, generator=generator, device=latent.device)
                latent = sigma_next[:, None, None, None] * noise + (1.0 - sigma_next[:, None, None, None]) * latent
                k += 1
                try:
                    next_scale = self.scales[k]
                except IndexError:
                    next_scale = 128

        return latent
    ## ---------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------