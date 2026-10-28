"""The audited native sweep's deterministic, spaced pixel-space DDIM sampler.

Preserves its terminal alpha_bar[0] convention; no new sampler formulation.
"""
import numpy as np
import torch


def ddim_times(t0, steps):
    if not 1 <= steps <= t0:
        raise ValueError(f'{steps} steps exceed starting timestep {t0}')
    current = np.rint(np.linspace(t0, 1, steps)).astype(np.int64)
    if len(np.unique(current)) != steps or not np.all(current[:-1] > current[1:]):
        raise ValueError('DDIM spacing must be strictly decreasing')
    previous = np.concatenate((current[1:], np.array([0], dtype=np.int64)))
    return current.tolist(), previous.tolist()


@torch.no_grad()
def spaced_ddim(model, scheduler, condition, strength, steps, guidance_scale):
    if not 0 < strength < 1 or guidance_scale < 1:
        raise ValueError('Unexpected sampler strength or guidance')
    device = condition.device
    t0 = max(1, int(np.clip(strength, 1e-6, .9999) * (scheduler.T - 1)))
    current, previous = ddim_times(t0, steps)
    start = torch.full((condition.shape[0],), t0, dtype=torch.long, device=device)
    x = scheduler.q_sample(condition, start, torch.randn_like(condition))
    zero_condition = torch.zeros_like(condition)
    for ti, si in zip(current, previous):
        t = torch.full((condition.shape[0],), ti, dtype=torch.long, device=device)
        cond_eps = model(x, condition, t)
        if guidance_scale == 1.0:
            epsilon = cond_eps
        else:
            uncond_eps = model(x, zero_condition, t)
            epsilon = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        abar_t = scheduler.acp_t[t][:, None, None, None]
        s = torch.full_like(t, si)
        abar_s = scheduler.acp_t[s][:, None, None, None]
        x0 = (x - torch.sqrt(1 - abar_t) * epsilon) / (torch.sqrt(abar_t) + 1e-8)
        x = torch.sqrt(abar_s) * x0 + torch.sqrt(1 - abar_s) * epsilon
        if not torch.isfinite(x).all():
            raise FloatingPointError(f'Non-finite DDIM state at timestep {ti}')
    return x.clamp(-1, 1), current
