from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from src.env import device


class PretrainedLPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        weights = tv_models.SqueezeNet1_1_Weights.DEFAULT
        net = tv_models.squeezenet1_1(weights=weights).features.eval()
        for p in net.parameters():
            p.requires_grad = False
        self.slices = nn.ModuleList([net[:2], net[2:5], net[5:8], net[8:10]])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        x_01 = torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)
        x_up = F.interpolate(x_01, size=(64, 64), mode="bilinear", align_corners=False)
        return (x_up - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        h_p = self.normalize_input(pred)
        h_t = self.normalize_input(target)
        dist = torch.zeros(pred.size(0), device=pred.device)
        for sl in self.slices:
            h_p = sl(h_p)
            h_t = sl(h_t)
            norm_p = h_p / (torch.sqrt(torch.sum(h_p ** 2, dim=1, keepdim=True)) + 1e-10)
            norm_t = h_t / (torch.sqrt(torch.sum(h_t ** 2, dim=1, keepdim=True)) + 1e-10)
            dist = dist + (norm_p - norm_t).pow(2).sum(dim=1).mean(dim=[1, 2])
        return dist / len(self.slices)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    x = torch.clamp((pred + 1.0) * 0.5, 0.0, 1.0)
    y = torch.clamp((target + 1.0) * 0.5, 0.0, 1.0)
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)

    pad = window_size // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=3)
    mu_y = F.conv2d(y, window, padding=pad, groups=3)
    mu_x_sq, mu_y_sq, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sigma_x_sq = torch.clamp(F.conv2d(x * x, window, padding=pad, groups=3) - mu_x_sq, min=0.0)
    sigma_y_sq = torch.clamp(F.conv2d(y * y, window, padding=pad, groups=3) - mu_y_sq, min=0.0)
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=3) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return ssim_map.mean(dim=[1, 2, 3])


class FSQ(nn.Module):
    def __init__(self, levels: List[int] = [5, 5, 5]):
        super().__init__()
        self.levels = torch.tensor(levels, dtype=torch.int32)
        self.codebook_size = int(np.prod(levels))
        basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.long)
        self.register_buffer("basis", basis)
        self.register_buffer("half_l", (self.levels.float() - 1.0) / 2.0)

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ste magic: roundeth in forward, passeth gradients straight through in backward
        z_p = z.permute(0, 2, 3, 1)
        half_l = self.half_l.to(z.device)
        z_b = torch.tanh(z_p) * half_l
        z_r = z_b + (torch.round(z_b) - z_b).detach()
        z_int = torch.round(z_b + half_l).long()
        indices = (z_int * self.basis.to(z.device)).sum(dim=-1)
        z_q = (z_r / half_l).permute(0, 3, 1, 2)
        return z_q, indices.reshape(z.size(0), -1)

    def indices_to_codes(self, indices: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B = indices.size(0)
        idx = indices.reshape(B, H, W, 1)
        z_int = (idx // self.basis.to(indices.device)) % self.levels.to(indices.device)
        half_l = self.half_l.to(indices.device)
        return ((z_int.float() - half_l) / half_l).permute(0, 3, 1, 2)


class DualCompressiveVQGAN(nn.Module):
    def __init__(self, fsq_levels=[5, 5, 5], ch: int = 64):
        super().__init__()
        d = len(fsq_levels)
        self.fsq_c = FSQ(fsq_levels)
        self.fsq_v = FSQ(fsq_levels)
        self.K = self.fsq_c.codebook_size

        self.E_c = nn.Sequential(
            nn.Conv2d(3, ch, 4, stride=2, padding=1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 4, stride=2, padding=1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, d, 1)
        )
        self.post_c = nn.Conv2d(d, ch, 1)
        self.D_c = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.ConvTranspose2d(ch, 3, 4, stride=2, padding=1), nn.Tanh()
        )

        self.E_v = nn.Sequential(
            nn.Conv2d(3, ch, 4, stride=2, padding=1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 4, stride=2, padding=1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 4, stride=2, padding=1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, d, 1)
        )
        self.post_v = nn.Sequential(
            nn.ConvTranspose2d(d, ch, 4, stride=2, padding=1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.SiLU()
        )
        # 2d pos emb + gated cross-attn lest the static background devour our red agent
        self.pos_emb = nn.Parameter(torch.randn(1, 64, ch) * 0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim=ch, num_heads=4, batch_first=True)
        self.norm_q = nn.LayerNorm(ch)
        self.norm_kv = nn.LayerNorm(ch)
        self.ctx_gate = nn.Parameter(torch.tensor(0.1))

        self.D_v = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1), nn.GroupNorm(8, ch), nn.SiLU(),
            nn.ConvTranspose2d(ch, 3, 4, stride=2, padding=1), nn.Tanh()
        )

    def encode_context(self, s_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_q_c, idx_c = self.fsq_c.quantize(self.E_c(s_c))
        return self.post_c(z_q_c), idx_c

    def encode_dynamics(self, s_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fsq_v.quantize(self.E_v(s_t))

    def decode_dynamics_codes(self, z_q_v: torch.Tensor, feat_c: torch.Tensor) -> torch.Tensor:
        feat_v = self.post_v(z_q_v)
        B, C, H, W = feat_v.shape
        q = self.norm_q(feat_v.flatten(2).transpose(1, 2) + self.pos_emb)
        kv = self.norm_kv(feat_c.flatten(2).transpose(1, 2) + self.pos_emb)
        attn_out, _ = self.cross_attn(q, kv, kv)
        fused = feat_v + torch.tanh(self.ctx_gate) * attn_out.transpose(1, 2).reshape(B, C, H, W)
        return self.D_v(fused)

    def decode_dynamics_indices(self, idx_v: torch.Tensor, feat_c: torch.Tensor) -> torch.Tensor:
        z_q_v = self.fsq_v.indices_to_codes(idx_v, 4, 4)
        return self.decode_dynamics_codes(z_q_v, feat_c)

    def forward(self, s_c: torch.Tensor, s_t: torch.Tensor):
        feat_c, _ = self.encode_context(s_c)
        recon_c = self.D_c(feat_c)
        z_q_v, _ = self.encode_dynamics(s_t)
        recon_t = self.decode_dynamics_codes(z_q_v, feat_c)
        return recon_c, recon_t


class UnifiedWorldSequenceBuilder:
    def __init__(self, K: int = 125, bins: int = 256):
        self.K = K
        self.bins = bins
        self.off_dyn = K
        self.off_bin = K * 2
        self.vocab_size = K * 2 + bins + 2

    def quantize_vec(self, x: torch.Tensor, v_min: float, v_max: float) -> torch.Tensor:
        norm = torch.clamp((x - v_min) / (v_max - v_min), 0.0, 1.0)
        return torch.round(norm * (self.bins - 1)).long() + self.off_bin

    def dequantize_vec(self, tokens: torch.Tensor, v_min: float, v_max: float) -> torch.Tensor:
        bins = torch.clamp(tokens - self.off_bin, 0, self.bins - 1).float()
        return v_min + (bins / (self.bins - 1.0)) * (v_max - v_min)

    @torch.no_grad()
    def build_multistep_eq8(self, frames: torch.Tensor, proprios: torch.Tensor, actions: torch.Tensor, vq: DualCompressiveVQGAN):
        # ip -> raw frames + proprio + act; op -> flat token stream of len 144
        _, idx_c = vq.encode_context(frames[:, 0])
        parts = [idx_c]
        for t in range(1, 5):
            _, idx_v = vq.encode_dynamics(frames[:, t])
            p_tok = self.quantize_vec(proprios[:, t], 0.10, 0.90)
            a_tok = self.quantize_vec(actions[:, t - 1], -0.15, 0.15)
            parts.extend([idx_v + self.off_dyn, p_tok, a_tok])
        return torch.cat(parts, dim=1)

    def get_multistep_target_mask(self, device=device) -> torch.Tensor:
        mask = torch.zeros(143, dtype=torch.bool, device=device)
        cursor = 84
        for _ in range(3):
            mask[cursor - 1 : cursor - 1 + 18] = True
            cursor += 20
        return mask


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_rope(dim: int, max_len: int, theta: float = 10000.0, device=device):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(max_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, H, L, D = x.shape
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    c = cos[:L, :].unsqueeze(0).unsqueeze(0)
    s = sin[:L, :].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class LLaMABlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = RMSNorm(dim)
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = apply_rope(q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2), cos, sin)
        k = apply_rope(k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2), cos, sin)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(out.transpose(1, 2).contiguous().view(B, L, D))
        h2 = self.norm2(x)
        return x + self.w2(F.silu(self.w1(h2)) * self.w3(h2))


class MicroLLaMA(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 192, n_layers: int = 4, n_heads: int = 6, max_len: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([LLaMABlock(dim, n_heads, dim * 4) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        cos, sin = precompute_rope(dim // n_heads, max_len, device=device)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        return self.lm_head(self.norm(x))