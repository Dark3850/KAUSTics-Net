"""Isolated epsilon/velocity pilot math; frozen historical code stays untouched."""
import numpy as np
import torch
from .model import gaussian_blur, chroma_uv
from .losses import (loss_terms, mean_per_image, normalized_lab,
                                ssim_per_image)
from .spaced_ddim import ddim_times, spaced_ddim


def coefficients(scheduler, timestep):
    return (scheduler.sqrt_acp_t[timestep][:, None, None, None],
            scheduler.sqrt_om_acp_t[timestep][:, None, None, None])


def velocity_target(scheduler, target, noise, timestep):
    a, b = coefficients(scheduler, timestep)
    return a * noise - b * target


def predictions(scheduler, xt, output, timestep, prediction_type):
    output = output.float()
    if prediction_type == 'epsilon':
        return scheduler.predict_x0(xt, output, timestep), output
    if prediction_type != 'v_prediction':
        raise ValueError(prediction_type)
    a, b = coefficients(scheduler, timestep)
    # No division by sqrt(alpha_bar): these are an orthogonal rotation.
    return a * xt - b * output, b * xt + a * output


def pilot_loss_terms(scheduler, target, xt, output, noise, timestep,
                     prediction_type='epsilon', rgb_loss='l1'):
    if rgb_loss not in ('l1', 'mse'):
        raise ValueError(rgb_loss)
    if prediction_type == 'epsilon' and rgb_loss == 'l1':
        terms = loss_terms(scheduler, target, xt, output, noise, timestep, .2, .2)
        return {'prediction': terms['epsilon'],
                **{k: v for k, v in terms.items() if k != 'epsilon'}}
    reconstructed, _ = predictions(scheduler, xt, output, timestep, prediction_type)
    core_target = (noise if prediction_type == 'epsilon' else
                   velocity_target(scheduler, target, noise, timestep))
    difference = reconstructed - target
    up, vp = chroma_uv(reconstructed * .5 + .5)
    ug, vg = chroma_uv(target * .5 + .5)
    rgb = (reconstructed * .5 + .5).clamp(0, 1)
    reference = target * .5 + .5
    return dict(
        prediction=.2 * mean_per_image((output.float() - core_target).square()),
        rgb=.8 * mean_per_image(difference.abs() if rgb_loss == 'l1' else difference.square()),
        lowfreq=.2 * mean_per_image((gaussian_blur(reconstructed, 21, 5.) -
                                     gaussian_blur(target, 21, 5.)).abs()),
        chroma=.3 * (mean_per_image((up-ug).abs()) + mean_per_image((vp-vg).abs())),
        ssim=.2 * (1 - ssim_per_image(rgb, reference)),
        lab=.2 * mean_per_image((normalized_lab(gaussian_blur(rgb, 21, 5.)) -
                                normalized_lab(gaussian_blur(reference, 21, 5.))).abs()))


def pilot_objective(terms):
    # Preserve the original D addition order in the paired reference arm.
    return (terms['prediction'] +
            sum(v for k, v in terms.items() if k != 'prediction')).mean()


@torch.no_grad()
def pilot_ddim(model, scheduler, condition, strength, steps, guidance_scale,
               prediction_type='epsilon', endpoint='legacy'):
    if prediction_type not in ('epsilon', 'v_prediction'):
        raise ValueError(prediction_type)
    if endpoint not in ('legacy', 'clean'):
        raise ValueError(endpoint)
    if not 0 < strength < 1 or guidance_scale < 1:
        raise ValueError('Unexpected sampler strength or guidance')
    if prediction_type == 'epsilon' and endpoint == 'legacy':
        return spaced_ddim(model, scheduler, condition, strength, steps, guidance_scale)
    t0 = max(1, int(np.clip(strength, 1e-6, .9999) * (scheduler.T - 1)))
    current, previous = ddim_times(t0, steps)
    start = torch.full((len(condition),), t0, dtype=torch.long, device=condition.device)
    x = scheduler.q_sample(condition, start, torch.randn_like(condition))
    null = torch.zeros_like(condition)
    for position, (ti, si) in enumerate(zip(current, previous)):
        t = torch.full_like(start, ti)
        output = model(x, condition, t)
        if guidance_scale != 1.:
            unconditioned = model(x, null, t)
            output = unconditioned + guidance_scale * (output - unconditioned)
        if prediction_type == 'epsilon':
            # Bitwise-preserve every intermediate legacy DDIM operation;
            # the endpoint experiment changes only the final destination.
            abar_t = scheduler.acp_t[t][:, None, None, None]
            epsilon = output
            x0 = (x - torch.sqrt(1 - abar_t) * epsilon) / (torch.sqrt(abar_t) + 1e-8)
        else:
            x0, epsilon = predictions(scheduler, x, output, t, prediction_type)
        if endpoint == 'clean' and position == len(current) - 1:
            # The final destination has alpha_bar=1, not alpha_bar[0].
            x = x0
        else:
            abar_s = scheduler.acp_t[torch.full_like(start, si)][:, None, None, None]
            x = torch.sqrt(abar_s) * x0 + torch.sqrt(1 - abar_s) * epsilon
        if not torch.isfinite(x).all():
            raise FloatingPointError(f'Nonfinite DDIM state at timestep {ti}')
    return x.clamp(-1, 1), current
