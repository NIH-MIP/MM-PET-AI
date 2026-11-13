"""MM-PET-AI network (inference-only).

Asymmetric PET and CT ViT encoders, bidirectional cross-modality context
exchange at depths 3/6/9/12, a PET-local convolutional branch, and a shared
UNETR decoder with scan-level lesion-presence and tracer heads.

Parameter names match the training implementation so released checkpoints
load with ``strict=True``. Modules used only during training (deep
supervision, organ heads, the small-lesion auxiliary head, and the unused
per-encoder UNETR skip blocks) are omitted; see ``weights.TRAINING_ONLY_PREFIXES``.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets import ViT

from .config import MODEL_KWARGS

_NORM = "instance"


# ---------------------------------------------------------------------------
# Cross-modality context exchange
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Pre-normalized multi-head self-attention over a token sequence (B, N, C)."""

    def __init__(self, dim: int, heads: int, dim_head: int = 64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        b, n, _ = x.shape
        q, k, v = (
            t.view(b, n, self.heads, self.dim_head).transpose(1, 2)
            for t in self.to_qkv(x).chunk(3, dim=-1)
        )
        attn = (torch.matmul(q, k.transpose(-1, -2)) * self.scale).softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, self.heads * self.dim_head)
        return self.to_out(out)


class Transformer(nn.Module):
    """One residual transformer block (attention + feed-forward)."""

    def __init__(self, dim: int, heads: int, mlp_dim: int, dim_head: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                nn.Sequential(nn.LayerNorm(dim), FeedForward(dim, mlp_dim)),
            ])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class CrossModalityAdapter(nn.Module):
    """Prepend the opposite modality's global token, attend, keep spatial tokens.

    ``e`` is the PET feature map (B, Ce, D, H, W) and ``r`` the CT feature map
    (B, Cr, D, H, W) on the same token grid.
    """

    def __init__(self, dim_e: int, dim_r: int):
        super().__init__()
        if dim_e % 64 or dim_r % 64:
            raise ValueError("Embedding dimensions must be divisible by 64.")
        self.dim_e, self.dim_r = dim_e, dim_r
        self.transformer_e = Transformer(dim_e, heads=dim_e // 64, mlp_dim=dim_e * 2)
        self.transformer_r = Transformer(dim_r, heads=dim_r // 64, mlp_dim=dim_r * 2)
        self.norm_e = nn.LayerNorm(dim_e)
        self.norm_r = nn.LayerNorm(dim_r)
        self.linear_e_to_r = nn.Linear(dim_e, dim_r)
        self.linear_r_to_e = nn.Linear(dim_r, dim_e)

    def forward(self, e: torch.Tensor, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, ce, d, h, w = e.shape
        cr = r.shape[1]
        e_tok = e.flatten(2).transpose(1, 2)  # (B, N, Ce)
        r_tok = r.flatten(2).transpose(1, 2)  # (B, N, Cr)

        e_global = self.norm_e(e_tok).mean(dim=1)
        r_global = self.norm_r(r_tok).mean(dim=1)
        e_for_r = self.linear_e_to_r(e_global).unsqueeze(1)
        r_for_e = self.linear_r_to_e(r_global).unsqueeze(1)

        r_out = self.transformer_r(torch.cat([e_for_r, r_tok], dim=1))[:, 1:, :]
        e_out = self.transformer_e(torch.cat([r_for_e, e_tok], dim=1))[:, 1:, :]
        e_out = e_out.transpose(1, 2).reshape(b, ce, d, h, w)
        r_out = r_out.transpose(1, 2).reshape(b, cr, d, h, w)
        return e_out, r_out


# ---------------------------------------------------------------------------
# Convolutional blocks and the PET-local branch
# ---------------------------------------------------------------------------

def _gn_groups(num_channels: int, max_groups: int = 32) -> int:
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        g = _gn_groups(out_ch)
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride, 1, bias=True),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=True),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PETLocalDetailEncoder(nn.Module):
    """PET-only convolutional features at full, 1/2, and 1/4 resolution."""

    def __init__(self, base_ch: int = 16):
        super().__init__()
        self.stem = ConvBlock3D(1, base_ch)
        self.down1 = ConvBlock3D(base_ch, base_ch * 2, stride=2)
        self.down2 = ConvBlock3D(base_ch * 2, base_ch * 4, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p0 = self.stem(x)
        p1 = self.down1(p0)
        p2 = self.down2(p1)
        return p0, p1, p2


# ---------------------------------------------------------------------------
# Modality-specific ViT encoder
# ---------------------------------------------------------------------------

class ViTEncoder(nn.Module):
    """MONAI ViT plus a shallow full-resolution convolutional feature."""

    def __init__(
        self,
        img_size: Sequence[int],
        patch_size: Sequence[int],
        feature_size: int,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.feat_size = tuple(i // p for i, p in zip(img_size, patch_size))
        self.vit = ViT(
            in_channels=1,
            img_size=tuple(img_size),
            patch_size=tuple(patch_size),
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=12,
            num_heads=num_heads,
            proj_type="conv",
            classification=False,
            dropout_rate=0.0,
            spatial_dims=3,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3, in_channels=1, out_channels=feature_size,
            kernel_size=3, stride=1, norm_name=_NORM, res_block=True,
        )

    def _to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        b, _, c = tokens.shape
        return tokens.view(b, *self.feat_size, c).permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z12, hidden = self.vit(x)
        return {
            "x0": self.encoder1(x),
            "z3": self._to_grid(hidden[2]),
            "z6": self._to_grid(hidden[5]),
            "z9": self._to_grid(hidden[8]),
            "z12": self._to_grid(z12),
        }


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def _pr_up(in_ch: int, out_ch: int, num_layer: int) -> UnetrPrUpBlock:
    return UnetrPrUpBlock(
        spatial_dims=3, in_channels=in_ch, out_channels=out_ch, num_layer=num_layer,
        kernel_size=3, stride=1, upsample_kernel_size=2, norm_name=_NORM,
        conv_block=True, res_block=True,
    )


def _up(in_ch: int, out_ch: int) -> UnetrUpBlock:
    return UnetrUpBlock(
        spatial_dims=3, in_channels=in_ch, out_channels=out_ch, kernel_size=3,
        upsample_kernel_size=2, norm_name=_NORM, res_block=True,
    )


class MMPETAINet(nn.Module):
    """Input (B, 2, 192, 192, 192) with channel 0 = PET and channel 1 = CT.

    Returns ``(seg_logits, lesion_logits, tracer_logits)`` with shapes
    (B, 1, D, H, W), (B, 1), and (B, 3).
    """

    def __init__(
        self,
        img_size: Sequence[int],
        patch_size: Sequence[int],
        feature_size: int,
        hidden_size_pet: int,
        mlp_dim_pet: int,
        num_heads_pet: int,
        hidden_size_ct: int,
        mlp_dim_ct: int,
        num_heads_ct: int,
        pet_local_base_ch: int,
    ):
        super().__init__()
        self.img_size = tuple(int(v) for v in img_size)
        fs, hp, hc, lc = feature_size, hidden_size_pet, hidden_size_ct, pet_local_base_ch

        self.encoder_pet = ViTEncoder(img_size, patch_size, fs, hp, mlp_dim_pet, num_heads_pet)
        self.encoder_ct = ViTEncoder(img_size, patch_size, fs, hc, mlp_dim_ct, num_heads_ct)

        self.pet_local_encoder = PETLocalDetailEncoder(lc)
        self.pet_local_proj0 = nn.Conv3d(lc, fs, kernel_size=1, bias=True)
        self.pet_local_proj1 = nn.Conv3d(lc * 2, fs * 2, kernel_size=1, bias=True)
        self.pet_local_proj2 = nn.Conv3d(lc * 4, fs * 4, kernel_size=1, bias=True)

        self.cross_z3 = CrossModalityAdapter(hp, hc)
        self.cross_z6 = CrossModalityAdapter(hp, hc)
        self.cross_z9 = CrossModalityAdapter(hp, hc)
        self.cross_z12 = CrossModalityAdapter(hp, hc)

        self.pet_reenc2 = _pr_up(hp, fs * 2, num_layer=2)
        self.pet_reenc3 = _pr_up(hp, fs * 4, num_layer=1)
        self.pet_reenc4 = _pr_up(hp, fs * 8, num_layer=0)
        self.ct_reenc2 = _pr_up(hc, fs * 2, num_layer=2)
        self.ct_reenc3 = _pr_up(hc, fs * 4, num_layer=1)
        self.ct_reenc4 = _pr_up(hc, fs * 8, num_layer=0)

        self.fuse0 = ConvBlock3D(fs * 2 + fs, fs)
        self.fuse1 = ConvBlock3D(fs * 4 + fs * 2, fs * 2)
        self.fuse2 = ConvBlock3D(fs * 8 + fs * 4, fs * 4)
        self.fuse3 = ConvBlock3D(fs * 16, fs * 8)

        self.bottleneck_pet_proj = nn.Conv3d(hp, hp, kernel_size=1, bias=True)
        self.bottleneck_ct_proj = nn.Conv3d(hc, hp, kernel_size=1, bias=True)
        self.fuse4 = ConvBlock3D(hp * 2, hp)

        self.decoder4 = _up(hp, fs * 8)
        self.decoder3 = _up(fs * 8, fs * 4)
        self.decoder2 = _up(fs * 4, fs * 2)
        self.decoder1 = _up(fs * 2, fs)
        self.final = nn.Conv3d(fs, 1, kernel_size=1)

        self.lesion_head = nn.Linear(hp, 1)
        self.tracer_head = nn.Linear(hp, 3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 5 or x.shape[1] != 2 or tuple(x.shape[2:]) != self.img_size:
            raise ValueError(f"Expected input (B, 2, *{self.img_size}), got {tuple(x.shape)}")
        x_pet, x_ct = x[:, :1], x[:, 1:2]

        pet = self.encoder_pet(x_pet)
        ct = self.encoder_ct(x_ct)
        p0, p1, p2 = self.pet_local_encoder(x_pet)

        z3p, z3c = self.cross_z3(pet["z3"], ct["z3"])
        z6p, z6c = self.cross_z6(pet["z6"], ct["z6"])
        z9p, z9c = self.cross_z9(pet["z9"], ct["z9"])
        z12p, z12c = self.cross_z12(pet["z12"], ct["z12"])

        s0 = self.fuse0(torch.cat([pet["x0"], ct["x0"], self.pet_local_proj0(p0)], dim=1))
        s1 = self.fuse1(torch.cat([self.pet_reenc2(z3p), self.ct_reenc2(z3c), self.pet_local_proj1(p1)], dim=1))
        s2 = self.fuse2(torch.cat([self.pet_reenc3(z6p), self.ct_reenc3(z6c), self.pet_local_proj2(p2)], dim=1))
        s3 = self.fuse3(torch.cat([self.pet_reenc4(z9p), self.ct_reenc4(z9c)], dim=1))

        b = self.fuse4(torch.cat([self.bottleneck_pet_proj(z12p), self.bottleneck_ct_proj(z12c)], dim=1))

        d = self.decoder4(b, s3)
        d = self.decoder3(d, s2)
        d = self.decoder2(d, s1)
        d = self.decoder1(d, s0)
        seg_logits = self.final(d)

        g = b.mean(dim=(2, 3, 4))
        return seg_logits, self.lesion_head(g), self.tracer_head(g)


def build_model() -> MMPETAINet:
    """Construct the released architecture (random weights)."""
    return MMPETAINet(**MODEL_KWARGS)
