"""
unet.py

COnditional u-net backbone for rectified flow model.
Operates in medvae latent space (64x64 with 4 channels)

The model receives:
    x_t: (B, 4, 64, 64) noisy interpolated latent
    t: (B, )  random timestep in [0, 1]
    cond: (B, 4, 64, 64) low-dose latent (which is the condition)

Outputs:
    v: (B, 4, 64, 64) predicted velocity

Architecture (base_channels=128, channel_mults=[1, 2, 4, 4]):
    Input: concat(x_t, cond) --> (B, 8, 64, 64) --> conv --> (B, 128, 64, 64)

    Encoder:
        Level 0: 64 x64, 128 ch (ResBlock x 2) --> downsample
        level 1: 32 x 32, 256 ch (resblock x 2) --> downsample
        level 2: 16 x16, 512 ch (resblock x 2) --> downsample
        level 3: 8 x8, 512 ch (resblock x 2) + attention
    
    Bottleneck: 8x8, 512 ch (resblock + attention + resblock)

    Decoder (with skip connections):
        level 3: 8x8, 512 ch (reslbock x 3 + attention) --> upsample
        level 2: 16 x16 512 ch (resblock x 3) --> upsample
        level 1: 32x 32, 256 ch(resblock x 3) --> upsample
        level 0: 64 x 64, 128 ch (reslbock x3) 

    output: GroupNorm --> SiLu --> conv --> (B, 4, 64, 64)

"""

import torch
import torch.nn as nn
import math

# ----- TIMESTEP EMBEDDING-------------
class SinusoidalTimestepEmbedding(nn.Module):
    """
    Maps scalar timestep t in [0, 1] to a vector. 
    t --> sinusoidal encoding --> mlp --> embedding vector
    """
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4

        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t[:, None] * 1000.0 * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)

#---------- BUILDING BLOCKS FOR U NET------------------
class ResBlock(nn.Module):
    """
    Residual block with timestep conditioning.
    x --> groupnorm --> silu --> conv --> (+t_emb) --> groupnorm --> silu --> dropout --> conv --> (+skip)
    """
    def __init__(self, in_ch, out_ch, t_emb_dim, dropout=.2):
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

class SelfAttention(nn.Module):
    """Single head self-attention on spatial feature maps"""
 
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5
 
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = (q.permute(0, 2, 1) @ k * self.scale).softmax(dim=-1)
        out = (v @ attn.permute(0, 2, 1)).reshape(B, C, H, W)
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
 
# ------------ ENCODER AND DECODER LEVEL ---------
class EncoderLevel(nn.Module):
    """One level of the encoder: N ResBlocks, optional attention, optional downsample"""
 
    def __init__(self, in_ch, out_ch, t_emb_dim, num_blocks=2,
                 use_attn=False, use_downsample=True, dropout=0.2):
        super().__init__()
 
        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()
 
        for i in range(num_blocks):
            self.blocks.append(ResBlock(
                in_ch if i == 0 else out_ch, out_ch, t_emb_dim, dropout
            ))
            self.attns.append(SelfAttention(out_ch) if use_attn else nn.Identity())
 
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
 
        return h, skips
 
 
class DecoderLevel(nn.Module):
    """One level of the decoder: (N+1) ResBlocks with skip concat, optional attention, optional upsample."""
 
    def __init__(self, in_ch, out_ch, skip_channels, t_emb_dim, num_blocks=3,
                 use_attn=False, use_upsample=True, dropout=0.1):
        super().__init__()
 
        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()
 
        for i in range(num_blocks):
            # Each block takes h + skip concatenated
            block_in = (in_ch if i == 0 else out_ch) + skip_channels[i]
            self.blocks.append(ResBlock(block_in, out_ch, t_emb_dim, dropout))
            self.attns.append(SelfAttention(out_ch) if use_attn else nn.Identity())
 
        self.upsample = Upsample(out_ch) if use_upsample else None
 
    def forward(self, x, skips, t_emb):
        h = x
        for block, attn in zip(self.blocks, self.attns):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
            h = attn(h)
 
        if self.upsample is not None:
            h = self.upsample(h)
 
        return h
    
# --------------- put together the full unet--------------------
class ConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        cond_channels=4,
        base_channels=128,
        channel_mults=(1, 2, 4, 4),
        num_res_blocks=2,
        dropout=0.1,
        attn_resolutions=(8,),
    ):
        super().__init__()

        self.in_channels = in_channels
        t_emb_dim = base_channels * 4

        self.time_embed = SinusoidalTimestepEmbedding(base_channels, t_emb_dim)
        self.input_conv = nn.Conv2d(
            in_channels + cond_channels, base_channels, 3, padding=1
        )

        # === Encoder ===
        self.encoder_levels = nn.ModuleList()
        ch = base_channels
        current_res = 64
        # Track skip channels per encoder level (block skips only, no downsample)
        level_out_channels = []

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            is_last = (level == len(channel_mults) - 1)

            self.encoder_levels.append(EncoderLevel(
                in_ch=ch, out_ch=out_ch, t_emb_dim=t_emb_dim,
                num_blocks=num_res_blocks,
                use_attn=(current_res in attn_resolutions),
                use_downsample=(not is_last), dropout=dropout,
            ))

            level_out_channels.append(out_ch)
            ch = out_ch
            if not is_last:
                current_res //= 2

        # --- bottleneck---
        self.bottleneck_res1 = ResBlock(ch, ch, t_emb_dim, dropout)
        self.bottleneck_attn = SelfAttention(ch)
        self.bottleneck_res2 = ResBlock(ch, ch, t_emb_dim, dropout)

        # ---Decoder (mirror of encoder)---
        self.decoder_levels = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            is_first = (level == 0)

            # Each decoder level pops num_res_blocks skips, all at out_ch channels
            skip_chs = [out_ch] * num_res_blocks

            self.decoder_levels.append(DecoderLevel(
                in_ch=ch, out_ch=out_ch, skip_channels=skip_chs,
                t_emb_dim=t_emb_dim, num_blocks=num_res_blocks,
                use_attn=(current_res in attn_resolutions),
                use_upsample=(not is_first), dropout=dropout,
            ))

            ch = out_ch
            if not is_first:
                current_res *= 2

        #-----Output-----
        self.output = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, x_t, t, cond):
        t_emb = self.time_embed(t)
        h = self.input_conv(torch.cat([x_t, cond], dim=1))

        # Encoder — collect all block skips into one flat list
        skips = []
        for level in self.encoder_levels:
            h, level_skips = level(h, t_emb)
            skips.extend(level_skips)

        # Bottleneck
        h = self.bottleneck_res1(h, t_emb)
        h = self.bottleneck_attn(h)
        h = self.bottleneck_res2(h, t_emb)

        # Decoder — each level pops what it needs
        for level in self.decoder_levels:
            h = level(h, skips, t_emb)

        return self.output(h)
 
# ---------- test!--------- (fake)
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    model = ConditionalUNet(
        in_channels=4,
        cond_channels=4,
        base_channels=128,
    ).to(device)
 
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"ConditionalUNet: {n_params:.1f}M parameters")
 
    # Test forward pass with medvae latent shapes
    B = 2
    x_t = torch.randn(B, 4, 64, 64).to(device)
    t = torch.rand(B).to(device)
    cond = torch.randn(B, 4, 64, 64).to(device)
 
    v = model(x_t, t, cond)
    print(f"\nInput:  x_t={x_t.shape}, t={t.shape}, cond={cond.shape}")
    print(f"Output: v={v.shape}")
    assert v.shape == x_t.shape, "Output shape must match input!"
 
    # Test gradient flow
    loss = v.sum()
    loss.backward()
    print(f"Gradient OK: {model.input_conv.weight.grad is not None}")
 
    print("\nBackbone OK!")

 

