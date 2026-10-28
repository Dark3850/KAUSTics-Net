"""Final sampler-aware fine-tuning objective (unchanged math)."""
from contextlib import nullcontext
import numpy as np
import torch
from .diffusion import predictions
from .spaced_ddim import ddim_times
from .losses import mean_per_image, normalized_lab, ssim_per_image
from .model import gaussian_blur

def sampled_restoration(model, scheduler, condition, *, strength=.95, steps=25,
                        guidance_scale=2., gradient_steps=4):
    if not 1 <= gradient_steps <= steps or not 0 < strength < 1:
        raise ValueError('Invalid truncated sampling protocol')
    t0 = max(1, int(np.clip(strength, 1e-6, .9999) * (scheduler.T - 1)))
    current, previous = ddim_times(t0, steps)
    start = torch.full((len(condition),), t0, dtype=torch.long, device=condition.device)
    x = scheduler.q_sample(condition, start, torch.randn_like(condition))
    null = torch.zeros_like(condition)
    for index, (ti, si) in enumerate(zip(current, previous)):
        active = index >= steps - gradient_steps
        if index == steps - gradient_steps:
            x = x.detach()
        with nullcontext() if active else torch.no_grad():
            t = torch.full_like(start, ti)
            output = model(x, condition, t)
            if guidance_scale != 1.:
                unconditioned = model(x, null, t)
                output = unconditioned + guidance_scale * (output - unconditioned)
            x0, epsilon = predictions(scheduler, x, output, t, 'v_prediction')
            abar_s = scheduler.acp_t[torch.full_like(start, si)][:, None, None, None]
            x = torch.sqrt(abar_s) * x0 + torch.sqrt(1 - abar_s) * epsilon
            if not torch.isfinite(x).all():
                raise FloatingPointError('Nonfinite sampler-aware state')
    return x


def sampled_loss(raw, target):
    rgb = (raw * .5 + .5).clamp(0, 1)
    reference = target * .5 + .5
    # RGB MSE remains UNCLIPPED, including outside the reporting range.
    return (.8 * mean_per_image((raw - target).square()) +
            .2 * (1 - ssim_per_image(rgb, reference)) +
            .2 * mean_per_image((normalized_lab(gaussian_blur(rgb, 21, 5.)) -
                                 normalized_lab(gaussian_blur(reference, 21, 5.))).abs())).mean()

