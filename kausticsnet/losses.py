"""FP32, per-image losses for the controlled 256 structure/colour study.

Existing reconstructed-RGB losses stay UNCLIPPED. Only the added SSIM/Lab
terms use RGB clipped to [0,1], as does restoration reporting.
"""
import torch
import torch.nn.functional as F
from .model import gaussian_blur, chroma_uv


def mean_per_image(value):
    return value.flatten(1).mean(1)


def ssim_per_image(prediction, target):
    """RGB SSIM: uniform 7x7 window, sample covariance, valid crop, range 1.

Matches skimage's default settings apart from unavoidable FP32 arithmetic.
"""
    if min(prediction.shape[-2:]) < 7 or prediction.shape != target.shape:
        raise ValueError('SSIM needs matching RGB images at least 7x7')
    mx, my = F.avg_pool2d(prediction, 7, 1), F.avg_pool2d(target, 7, 1)
    covariance_factor = 49. / 48.
    vx = ((F.avg_pool2d(prediction.square(), 7, 1) - mx.square()) * covariance_factor).clamp_min(0)
    vy = ((F.avg_pool2d(target.square(), 7, 1) - my.square()) * covariance_factor).clamp_min(0)
    cov = (F.avg_pool2d(prediction * target, 7, 1) - mx * my) * covariance_factor
    score = ((2 * mx * my + .01**2) * (2 * cov + .03**2) /
             ((mx.square() + my.square() + .01**2) * (vx + vy + .03**2)))
    return mean_per_image(score.clamp(-1, 1))


def normalized_lab(rgb):
    """Differentiable sRGB -> D65/2-degree CIELab, scaled L/100, a/128,b/128.

Guard cube-root's unused branch at black to avoid 0 * infinite derivatives.
"""
    rgb = rgb.float().clamp(0, 1)
    linear = torch.where(rgb > .04045, ((rgb + .055) / 1.055).pow(2.4), rgb / 12.92)
    matrix = rgb.new_tensor([[.412453, .357580, .180423],
                             [.212671, .715160, .072169],
                             [.019334, .119193, .950227]])
    xyz = torch.einsum('ij,njhw->nihw', matrix, linear)
    xyz = xyz / rgb.new_tensor([.95047, 1., 1.08883])[None, :, None, None]
    delta = 6. / 29.
    f = torch.where(xyz > delta**3, xyz.clamp_min(1e-8).pow(1./3.),
                    xyz / (3 * delta**2) + 4./29.)
    return torch.stack(((116 * f[:, 1] - 16) / 100,
                        500 * (f[:, 0] - f[:, 1]) / 128,
                        200 * (f[:, 1] - f[:, 2]) / 128), dim=1)


def loss_terms(scheduler, target, xt, prediction, noise, timestep,
               ssim_weight=0., lab_weight=0., blur_k=21, blur_sigma=5.):
    prediction = prediction.float()
    reconstructed = scheduler.predict_x0(xt, prediction, timestep)
    up, vp = chroma_uv(reconstructed * .5 + .5)
    ug, vg = chroma_uv(target * .5 + .5)
    terms = dict(epsilon=.2 * mean_per_image((prediction-noise).square()),
                 rgb=.8 * mean_per_image((reconstructed-target).abs()),
                 lowfreq=.2 * mean_per_image((gaussian_blur(reconstructed, blur_k, blur_sigma) -
                                             gaussian_blur(target, blur_k, blur_sigma)).abs()),
                 chroma=.3 * (mean_per_image((up-ug).abs()) + mean_per_image((vp-vg).abs())))
    zero = prediction.new_zeros(prediction.shape[0])
    terms['ssim'] = zero
    terms['lab'] = zero
    if ssim_weight or lab_weight:
        rgb, reference = (reconstructed * .5 + .5).clamp(0, 1), target * .5 + .5
        if ssim_weight:
            terms['ssim'] = ssim_weight * (1-ssim_per_image(rgb, reference))
        if lab_weight:
            # All three normalized channels constrain broad brightness AND chroma.
            pred_lab = normalized_lab(gaussian_blur(rgb, blur_k, blur_sigma))
            ref_lab = normalized_lab(gaussian_blur(reference, blur_k, blur_sigma))
            terms['lab'] = lab_weight * mean_per_image((pred_lab-ref_lab).abs())
    return terms


def objective(scheduler, terms, timestep, weighting='none'):
    if weighting == 'none':
        weight = torch.ones_like(terms['epsilon'])
    elif weighting == 'sqrt_alpha':
        # E_t[w(t)]=1 over the unchanged uniform training horizon, max ~1.6.
        weight = scheduler.sqrt_acp_t[timestep] / scheduler.sqrt_acp_t[1:].mean()
    else:
        raise ValueError(weighting)
    aux = sum(value for name, value in terms.items() if name != 'epsilon')
    return (terms['epsilon'] + weight * aux).mean()
