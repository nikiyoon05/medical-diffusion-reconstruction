"""
unet.py  (v2)

Conditional U-Net backbone for rectified flow — improved architecture.

Changes from v1:
    1. Multi-head self-attention (4 heads) at 8×8 AND 16×16
    2. Cross-attention for conditioning (Q=decoder, K/V=condition)
    3. 3 ResBlocks per level (was 2)
    4. Condition encoder — gives multi-scale condition features for cross-attn

The backbone receives:
    x_t:  (B, 4, 64, 64)   noisy interpolated latent
    t:  (B,) timestep in [0, 1]
    cond: (B, 4, 64, 64)   low-dose latent (condition)

Outputs:
    v: (B, 4, 64, 64)   predicted velocity
"""

import torch
import torch.nn as nn
import math


# -------------------
# Timestep Embedding
# ----------------
class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t[:, None] * 1000.0 * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


# --------------------------------------
# Building Blocks
# ------------------------------------

class ResBlock(nn.Module):
    """Residual block with timestep conditioning."""

    def __init__(self, in_ch, out_ch, t_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_ch))
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(torch.nn.functional.silu(self.norm2(h))))
        return h + self.skip(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention on spatial feature maps."""

    def __init__(self, channels, n_heads=4):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h).reshape(B, 3, self.n_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # q, k, v: (B, n_heads, head_dim, HW)

        # Used chatgpt for einsum
        attn = torch.einsum("bhdi,bhdj->bhij", q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bhij,bhdj->bhdi", attn, v)
        out = out.reshape(B, C, H, W)

        return x + self.proj(out)


class CrossAttention(nn.Module):
    """
    Cross-attention: Q from main features, K/V from condition features.
    Lets the model selectively attend to different parts of the condition.
    """

    def __init__(self, channels, cond_channels, n_heads=4):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(8, channels)
        self.norm_cond = nn.GroupNorm(8, cond_channels)

        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_kv = nn.Conv2d(cond_channels, channels * 2, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, cond):
        """
        x:    (B, C, H, W)  — main features
        cond: (B, C_cond, H, W) — condition features (same spatial size)
        """
        B, C, H, W = x.shape

        q = self.to_q(self.norm(x)).reshape(B, self.n_heads, self.head_dim, H * W)
        kv = self.to_kv(self.norm_cond(cond)).reshape(B, 2, self.n_heads, self.head_dim, H * W)
        k, v = kv[:, 0], kv[:, 1]

        attn = torch.einsum("bhdi,bhdj->bhij", q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bhij,bhdj->bhdi", attn, v)
        out = out.reshape(B, C, H, W)

        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(nn.functional.interpolate(x, scale_factor=2, mode="nearest"))


# --------------Condition Encoder----------------

class ConditionEncoder(nn.Module):
    """
    Lightweight encoder that produces multi-scale condition features
    for cross-attention at each decoder level.

    Input: (B, 4, 64, 64) condition latent
    Output: dict of {resolution: (B, channels, H, W)} at each level
    """

    def __init__(self, in_channels=4, base_channels=128, channel_mults=(1, 2, 4, 4)):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = base_channels
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.levels.append(nn.Sequential(
                nn.GroupNorm(8, ch),
                nn.SiLU(),
                nn.Conv2d(ch, out_ch, 3, padding=1),
            ))
            ch = out_ch

            if level < len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))

    def forward(self, cond):
        """Returns list of features from coarsest to finest resolution."""
        features = {}
        h = self.input_conv(cond)

        current_res = cond.shape[-1]  # 64

        for level, layer in enumerate(self.levels):
            h = layer(h)
            features[current_res] = h

            if level < len(self.downsamples):
                h = self.downsamples[level](h)
                current_res //= 2

        return features


# ----------------Encoder / Decoder Levels---------------------------
class EncoderLevel(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim, num_blocks=3,
                 use_attn=False, use_downsample=True, dropout=0.1, n_heads=4):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()

        for i in range(num_blocks):
            self.blocks.append(ResBlock(
                in_ch if i == 0 else out_ch, out_ch, t_emb_dim, dropout
            ))
            self.attns.append(
                MultiHeadSelfAttention(out_ch, n_heads) if use_attn
                else nn.Identity()
            )

        self.downsample = Downsample(out_ch) if use_downsample else None

    def forward(self, x, t_emb):
        skips = []
        h = x
        for block, attn in zip(self.blocks, self.attns):
            h = block(h, t_emb)
            h = attn(h)
            skips.append(h)

        if self.downsample is not None:
            h = self.downsample(h)
            # Don't append downsample to skips

        return h, skips


class DecoderLevel(nn.Module):
    def __init__(self, in_ch, out_ch, skip_channels, t_emb_dim, num_blocks=3,
                 use_attn=False, use_cross_attn=False, cond_channels=None,
                 use_upsample=True, dropout=0.1, n_heads=4):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.self_attns = nn.ModuleList()
        self.cross_attns = nn.ModuleList()

        for i in range(num_blocks):
            block_in = (in_ch if i == 0 else out_ch) + skip_channels[i]
            self.blocks.append(ResBlock(block_in, out_ch, t_emb_dim, dropout))

            self.self_attns.append(
                MultiHeadSelfAttention(out_ch, n_heads) if use_attn
                else nn.Identity()
            )
            self.cross_attns.append(
                CrossAttention(out_ch, cond_channels or out_ch, n_heads) if use_cross_attn
                else None
            )

        self.upsample = Upsample(out_ch) if use_upsample else None

    def forward(self, x, skips, t_emb, cond_features=None):
        h = x
        for block, self_attn, cross_attn in zip(
            self.blocks, self.self_attns, self.cross_attns
        ):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
            h = self_attn(h)

            if cross_attn is not None and cond_features is not None:
                h = cross_attn(h, cond_features)

        if self.upsample is not None:
            h = self.upsample(h)

        return h


# Full U-Net v2

class ConditionalUNet(nn.Module):
    """
    Improved Conditional U-Net for velocity prediction.

    Improvements over v1:
        - Multi-head self-attention (4 heads) at 8×8 and 16×16
        - Cross-attention for conditioning at 8×8 and 16×16
        - 3 ResBlocks per level
        - Condition encoder for multi-scale condition features
    """

    def __init__(
        self,
        in_channels=4,
        cond_channels=4,
        base_channels=128,
        channel_mults=(1, 2, 4, 4),
        num_res_blocks=3,
        dropout=0.1,
        attn_resolutions=(8, 16),
        cross_attn_resolutions=(8, 16),
        n_heads=4,
    ):
        super().__init__()

        self.in_channels = in_channels
        t_emb_dim = base_channels * 4

        self.time_embed = SinusoidalTimestepEmbedding(base_channels, t_emb_dim)

        # Condition encoder — produces multi-scale features for cross-attention
        self.cond_encoder = ConditionEncoder(
            in_channels=cond_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
        )

        # Input: concat x_t and condition, project to base_channels
        self.input_conv = nn.Conv2d(
            in_channels + cond_channels, base_channels, 3, padding=1
        )

        # ------Encoder-----
        self.encoder_levels = nn.ModuleList()
        ch = base_channels
        current_res = 64
        level_out_channels = []

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            is_last = (level == len(channel_mults) - 1)

            self.encoder_levels.append(EncoderLevel(
                in_ch=ch, out_ch=out_ch, t_emb_dim=t_emb_dim,
                num_blocks=num_res_blocks,
                use_attn=(current_res in attn_resolutions),
                use_downsample=(not is_last),
                dropout=dropout, n_heads=n_heads,
            ))

            level_out_channels.append(out_ch)
            ch = out_ch
            if not is_last:
                current_res //= 2

        # ----Bottleneck ===
        self.bottleneck_res1 = ResBlock(ch, ch, t_emb_dim, dropout)
        self.bottleneck_attn = MultiHeadSelfAttention(ch, n_heads)
        self.bottleneck_cross = CrossAttention(ch, ch, n_heads)
        self.bottleneck_res2 = ResBlock(ch, ch, t_emb_dim, dropout)

        # Decoder 
        self.decoder_levels = nn.ModuleList()
        # Track resolutions for cross-attention
        self.decoder_resolutions = []

        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            is_first = (level == 0)
            use_cross = (current_res in cross_attn_resolutions)

            skip_chs = [out_ch] * num_res_blocks

            # Condition channels at this resolution from condition encoder
            cond_ch_at_res = base_channels * mult

            self.decoder_levels.append(DecoderLevel(
                in_ch=ch, out_ch=out_ch, skip_channels=skip_chs,
                t_emb_dim=t_emb_dim, num_blocks=num_res_blocks,
                use_attn=(current_res in attn_resolutions),
                use_cross_attn=use_cross,
                cond_channels=cond_ch_at_res if use_cross else None,
                use_upsample=(not is_first),
                dropout=dropout, n_heads=n_heads,
            ))

            self.decoder_resolutions.append(current_res)
            ch = out_ch
            if not is_first:
                current_res *= 2

        # Output
        self.output = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, x_t, t, cond):
        """
        Args:
            x_t:  (B, 4, 64, 64)
            t:    (B,)
            cond: (B, 4, 64, 64)
        Returns:
            v:    (B, 4, 64, 64)
        """
        t_emb = self.time_embed(t)

        # Encode condition to multi-scale features
        cond_features = self.cond_encoder(cond)  # {64: ..., 32: ..., 16: ..., 8: ...}

        # Concat and project
        h = self.input_conv(torch.cat([x_t, cond], dim=1))

        # Encoder
        skips = []
        for level in self.encoder_levels:
            h, level_skips = level(h, t_emb)
            skips.extend(level_skips)

        # Bottleneck
        h = self.bottleneck_res1(h, t_emb)
        h = self.bottleneck_attn(h)
        # Cross-attend to condition at bottleneck resolution
        bottleneck_res = h.shape[-1]
        if bottleneck_res in cond_features:
            h = self.bottleneck_cross(h, cond_features[bottleneck_res])
        h = self.bottleneck_res2(h, t_emb)

        # Decoder
        for level, res in zip(self.decoder_levels, self.decoder_resolutions):
            # Get condition features at this resolution for cross-attention
            cf = cond_features.get(res, None)
            h = level(h, skips, t_emb, cond_features=cf)

        return self.output(h)



# Test

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConditionalUNet(
        in_channels=4,
        cond_channels=4,
        base_channels=128,
        num_res_blocks=3,
        attn_resolutions=(8, 16),
        cross_attn_resolutions=(8, 16),
        n_heads=4,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"ConditionalUNet v2: {n_params:.1f}M parameters")

    B = 2
    x_t = torch.randn(B, 4, 64, 64).to(device)
    t = torch.rand(B).to(device)
    cond = torch.randn(B, 4, 64, 64).to(device)

    v = model(x_t, t, cond)
    print(f"Input:  x_t={x_t.shape}, t={t.shape}, cond={cond.shape}")
    print(f"Output: v={v.shape}")
    assert v.shape == x_t.shape, "Output shape must match input!"

    loss = v.sum()
    loss.backward()
    print(f"Gradient OK: {model.input_conv.weight.grad is not None}")

    print("\nBackbone v2 OK!")