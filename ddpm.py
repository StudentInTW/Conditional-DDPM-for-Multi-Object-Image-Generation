
import math
import torch


class DDPMScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        schedule_type="cosine",
        beta_start=1e-4,
        beta_end=0.02,
        cosine_s=0.008,
        device="cuda"
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        self.schedule_type = schedule_type

        if schedule_type == "linear":
            self.betas = torch.linspace(
                beta_start,
                beta_end,
                num_timesteps,
                device=device
            )

        elif schedule_type == "cosine":
            self.betas = self._cosine_beta_schedule(
                num_timesteps=num_timesteps,
                s=cosine_s,
                device=device
            )

        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        # alpha_t = 1 - beta_t
        self.alphas = 1.0 - self.betas

        # alpha_bar_t = product_{i=0}^{t} alpha_i
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # alpha_bar_{t-1}, with alpha_bar_{-1} = 1
        self.alpha_bars_prev = torch.cat([
            torch.ones(1, device=device),
            self.alpha_bars[:-1]
        ], dim=0)

        # Common precomputed terms
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior variance:
        # beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )

        # Avoid numerical issues at t = 0
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

    def _cosine_beta_schedule(self, num_timesteps, s=0.008, device="cuda"):
        """
        Cosine noise schedule.

        First define alpha_bar_t using a squared cosine curve.
        Then derive beta_t from consecutive alpha_bar values.
        """
        steps = num_timesteps + 1

        x = torch.linspace(
            0,
            num_timesteps,
            steps,
            device=device
        )

        alpha_bars = torch.cos(
            ((x / num_timesteps + s) / (1 + s)) * math.pi * 0.5
        ) ** 2

        alpha_bars = alpha_bars / alpha_bars[0]

        betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.clamp(betas, min=1e-8, max=0.999)

        return betas

    def _extract(self, values, t, x_shape):
        """
        Extract values[t] and reshape to [B, 1, 1, 1],
        so it can broadcast with image tensor [B, C, H, W].
        """
        batch_size = t.shape[0]
        out = values.gather(0, t)
        return out.view(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        """
        Forward diffusion:

            x_t = sqrt(alpha_bar_t) * x0
                  + sqrt(1 - alpha_bar_t) * noise

        Args:
            x0: clean images, [B, C, H, W], normalized to [-1, 1]
            t: timestep indices, [B]
            noise: Gaussian noise, same shape as x0

        Returns:
            xt: noisy image
            noise: actual noise used
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self._extract(
            self.sqrt_alpha_bars, t, x0.shape
        )

        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alpha_bars, t, x0.shape
        )

        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

        return xt, noise

    @torch.no_grad()
    def p_sample(self, model, xt, t, labels, guidance_scale=1.0):
        """
        One reverse denoising step:

            x_t -> x_{t-1}

        The model predicts noise epsilon_theta(x_t, t, c).
        """
        
        if abs(guidance_scale - 1.0) < 1e-8:
            pred_noise = model(xt, t, labels)
        else:
            zero_labels = torch.zeros_like(labels)

            pred_noise_uncond = model(xt, t, zero_labels)
            pred_noise_cond = model(xt, t, labels)

            pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)


        beta_t = self._extract(self.betas, t, xt.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alpha_bars, t, xt.shape
        )
        sqrt_recip_alpha_t = self._extract(
            self.sqrt_recip_alphas, t, xt.shape
        )

        # DDPM mean:
        # mu_theta = 1 / sqrt(alpha_t) *
        #            (x_t - beta_t / sqrt(1 - alpha_bar_t) * epsilon_theta)
        model_mean = sqrt_recip_alpha_t * (
            xt - beta_t * pred_noise / sqrt_one_minus_alpha_bar_t
        )

        posterior_variance_t = self._extract(
            self.posterior_variance, t, xt.shape
        )

        noise = torch.randn_like(xt)

        # No noise when t = 0
        nonzero_mask = (t != 0).float().view(
            xt.shape[0],
            *((1,) * (len(xt.shape) - 1))
        )

        x_prev = model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model,
        labels,
        image_size=64,
        image_channels=3,
        return_intermediates=False,
        intermediate_steps=None,
         guidance_scale=1.0
    ):
        """
        Generate images from pure Gaussian noise.

        Args:
            model: trained ConditionalUNet
            labels: [B, 24] multi-hot condition vectors
            image_size: image resolution
            image_channels: usually 3
            return_intermediates: whether to save denoising process
            intermediate_steps: list of timesteps to record, e.g.
                                [999, 800, 600, 400, 200, 100, 50, 0]

        Returns:
            x: generated images, [B, 3, 64, 64]
            intermediates: optional list of intermediate images
        """
        model.eval()

        device = labels.device
        batch_size = labels.shape[0]

        x = torch.randn(
            batch_size,
            image_channels,
            image_size,
            image_size,
            device=device
        )

        intermediates = []

        if intermediate_steps is None:
            intermediate_steps = [999, 800, 600, 400, 200, 100, 50, 0]

        intermediate_steps = set(intermediate_steps)

        for step in reversed(range(self.num_timesteps)):
            t = torch.full(
                (batch_size,),
                step,
                device=device,
                dtype=torch.long
            )

            x = self.p_sample(model, x, t, labels, guidance_scale=guidance_scale)

            if return_intermediates and step in intermediate_steps:
                intermediates.append(x.detach().cpu())

        if return_intermediates:
            return x, intermediates

        return x