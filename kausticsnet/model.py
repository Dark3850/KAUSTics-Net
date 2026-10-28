"""Exact model/schedule functions extracted from the frozen revision implementation."""
import math
import random
from typing import Optional
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_uint8_img(x: torch.Tensor) -> np.ndarray:
    x = (x.detach().cpu().clamp(-1, 1) * 0.5 + 0.5)
    return (x * 255.0).round().byte().permute(1, 2, 0).numpy()


def sinusoidal_positional_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(1, half))
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, use_cond_film: bool = True, cond_ch: int = None):
        super().__init__()
        self.use_cond_film = use_cond_film
        self.skip_proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, 2 * in_ch))
        if use_cond_film:
            if cond_ch is None:
                cond_ch = in_ch
            self.cond_film = nn.Linear(cond_ch, 2 * in_ch)

    def forward(self, x, t_emb, cond_feat: Optional[torch.Tensor] = None):
        h = self.norm1(x)
        scale_t, shift_t = self.time_mlp(t_emb).chunk(2, dim=1)
        h = h * (1 + scale_t[:, :, None, None]) + shift_t[:, :, None, None]
        if self.use_cond_film and cond_feat is not None:
            g = F.adaptive_avg_pool2d(cond_feat, 1).flatten(1)
            gscale, gshift = self.cond_film(g).chunk(2, dim=1)
            h = h * (1 + gscale[:, :, None, None]) + gshift[:, :, None, None]
        h = F.silu(h); h = self.conv1(h)
        h = self.norm2(h); h = F.silu(h); h = self.conv2(h)
        return self.skip_proj(x) + h


def _nearest_pow2(n: int) -> int:
    if n <= 1: return 1
    lo = 1 << (n.bit_length() - 1); hi = lo << 1
    return lo if (n - lo) < (hi - n) else hi


def _make_pyramid_sizes(image_size: int, levels: int = 6):
    sz = [image_size]; cur = image_size
    for _ in range(levels - 1):
        cur = max(16, _nearest_pow2(max(1, int(round(cur / 2)))))
        sz.append(cur)
    return sz


class CondEncoder(nn.Module):
    def __init__(self, in_ch=3, chs=(16, 32, 64, 128, 256), image_size=512, levels=6):
        super().__init__()
        self.sz = _make_pyramid_sizes(image_size, levels=levels)
        blocks = []
        cur_ch = in_ch
        for c in chs:
            blocks.append(nn.Sequential(
                nn.Conv2d(cur_ch, c, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(c, c, 3, padding=1),
                nn.SiLU(),
            ))
            cur_ch = c
        self.blocks = nn.ModuleList(blocks)
        self.channels = chs

    def forward(self, x):
        cond_feats = []
        h = F.interpolate(x, size=self.sz[0], mode='bilinear', align_corners=False)
        for i, block in enumerate(self.blocks):
            h = block(h)
            cond_feats.append(h)
            if i < len(self.blocks) - 1:
                h = F.interpolate(h, size=self.sz[i+1], mode='bilinear', align_corners=False)
        return cond_feats


class SlimUNet(nn.Module):
    def __init__(self, time_emb_dim=256, image_size=512, use_film=True,
                 use_cross_attention=True, use_cond_encoder=True):
        super().__init__()
        self.image_size = image_size
        self.ch = [16, 32, 64, 128, 256]
        self.sz = _make_pyramid_sizes(image_size, levels=6)
        self.use_cross_attention = use_cross_attention
        self.use_cond_encoder = use_cond_encoder

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4), nn.SiLU(), nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # concat [x_noisy, x_cond]
        self.init_conv = nn.Conv2d(6, self.ch[0], 3, padding=1)
        self.cond_enc = CondEncoder(in_ch=3, chs=self.ch, image_size=image_size, levels=6) if use_cond_encoder else None

        self.down = nn.ModuleList([
            ResidualBlock(self.ch[i], self.ch[i+1], time_emb_dim, use_cond_film=use_film, cond_ch=self.ch[i])
            for i in range(len(self.ch) - 1)
        ])
        self.bottleneck = ResidualBlock(self.ch[-1], self.ch[-1], time_emb_dim, use_cond_film=use_film, cond_ch=self.ch[-1])

        self.indices_up = list(reversed(range(len(self.ch) - 1)))
        in_ch_up  = [self.ch[i+1] * 2 for i in self.indices_up]
        out_ch_up = [self.ch[i]       for i in self.indices_up]
        cond_up   = [self.ch[i]       for i in self.indices_up]
        self.up = nn.ModuleList([
            ResidualBlock(in_ch_u, out_ch_u, time_emb_dim, use_cond_film=use_film, cond_ch=cond_ch_u)
            for in_ch_u, out_ch_u, cond_ch_u in zip(in_ch_up, out_ch_up, cond_up)
        ])

        self.final_conv = nn.Conv2d(self.ch[0], 3, 1)
        self.cond_proj = nn.Conv2d(self.ch[-1], self.ch[-1], 1) if use_cross_attention else None
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.ch[-1], num_heads=1, batch_first=True) if use_cross_attention else None

    def apply_cross_attn(self, x, cond):
        B, C, H, W = x.shape
        q  = x.flatten(2).permute(0, 2, 1)
        kv = cond.flatten(2).permute(0, 2, 1)
        # Enables memory-efficient SDPA; no change to the attention definition.
        out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        return out.permute(0, 2, 1).view(B, C, H, W)

    def forward(self, x_noisy, x_cond, t):
        device = x_noisy.device
        if not torch.is_tensor(t):
            t = torch.full((x_noisy.size(0),), int(t), device=device, dtype=torch.long)
        else:
            t = t.to(device, dtype=torch.long).view(-1)
        t_emb = self.time_mlp(sinusoidal_positional_embedding(t, self.time_mlp[0].in_features).to(device))

        x_n = F.interpolate(x_noisy, size=self.sz[0], mode='bilinear', align_corners=False)
        x_c = F.interpolate(x_cond,  size=self.sz[0], mode='bilinear', align_corners=False)
        x = torch.cat([x_n, x_c], dim=1)
        x = self.init_conv(x)

        if self.use_cond_encoder:
            cond_feats = self.cond_enc(x_cond)
        else:
            # Fixed RGB pyramid isolates removal of the learned encoder while
            # retaining the FiLM and attention conditioning routes.
            cond_feats = [F.interpolate(x_cond, size=self.sz[i], mode='bilinear',
                          align_corners=False).repeat(1, (c + 2) // 3, 1, 1)[:, :c]
                          for i, c in enumerate(self.ch)]

        skips = []
        for i, block in enumerate(self.down):
            cf = cond_feats[i] if i < len(cond_feats) else None
            x = block(x, t_emb, cond_feat=cf)
            skips.append(x)
            x = F.interpolate(x, size=self.sz[i+1], mode='bilinear', align_corners=False)

        cf = cond_feats[-1] if len(cond_feats) else None
        x = self.bottleneck(x, t_emb, cond_feat=cf)
        if self.use_cross_attention:
            c = F.interpolate(cf, size=x.shape[-2:], mode='bilinear', align_corners=False)
            c = self.cond_proj(c)
            x = self.apply_cross_attn(x, c)

        for i, (block, skip, size) in enumerate(zip(self.up, reversed(skips), reversed(self.sz[:len(self.up)]))):
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            cf_up = cond_feats[len(self.ch)-2 - i] if i < len(self.ch)-1 else None
            x = block(x, t_emb, cond_feat=cf_up)

        x = F.interpolate(x, size=self.sz[0], mode='bilinear', align_corners=False)
        return self.final_conv(x)


def cosine_beta_schedule(T: int, s: float = 0.008) -> np.ndarray:
    t = np.linspace(0, T, T + 1, dtype=np.float64)
    f = np.cos(((t / T) + s) / (1.0 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-6, 0.999)


class DiffusionScheduler:
    def __init__(self, timesteps=1000, betas=None):
        self.T = timesteps
        self.betas = betas if betas is not None else cosine_beta_schedule(timesteps).astype(np.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1 - self.alphas_cumprod)

    def to_device(self, device):
        self.sqrt_acp_t = torch.from_numpy(self.sqrt_alphas_cumprod).to(device).float()
        self.sqrt_om_acp_t = torch.from_numpy(self.sqrt_one_minus_alphas_cumprod).to(device).float()
        self.acp_t = torch.from_numpy(self.alphas_cumprod).to(device).float()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_acp = self.sqrt_acp_t[t][:, None, None, None]
        sqrt_om  = self.sqrt_om_acp_t[t][:, None, None, None]
        return sqrt_acp * x0 + sqrt_om * noise

    def predict_x0(self, xt: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sqrt_acp = self.sqrt_acp_t[t][:, None, None, None]
        sqrt_om  = self.sqrt_om_acp_t[t][:, None, None, None]
        return (xt - sqrt_om * eps) / (sqrt_acp + 1e-8)


def gaussian_kernel1d(kernel_size=21, sigma=5.0, device="cpu", dtype=torch.float32):
    x = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    w = torch.exp(-(x**2) / (2 * sigma**2))
    w = (w / w.sum()).view(1, 1, -1)
    return w


def gaussian_blur(img, k=21, sigma=5.0):
    device, dtype = img.device, img.dtype
    w = gaussian_kernel1d(k, sigma, device, dtype)
    pad = (k // 2, k // 2)
    img = F.pad(img, (pad[0], pad[1], 0, 0), mode="reflect")
    img = F.conv2d(img, w.unsqueeze(2).repeat(img.size(1),1,1,1), groups=img.size(1))
    img = F.pad(img, (0, 0, pad[0], pad[1]), mode="reflect")
    img = F.conv2d(img, w.unsqueeze(3).repeat(img.size(1),1,1,1), groups=img.size(1))
    return img


def chroma_uv(img01):
    r, g, b = img01[:,0:1], img01[:,1:2], img01[:,2:3]
    U = -0.169*r - 0.331*g + 0.5*b + 0.5
    V =  0.5*r  - 0.419*g - 0.081*b + 0.5
    return U, V

