"""
pix2pix.py

Minimal pix2pix implementation for low-dose -->  full-dose CT denoising.
Based on "Image-to-Image Translation with Conditional Adversarial Networks"
(Isola et al., 2017).

Components:
    - UNetGenerator:      Encoder-decoder with skip connections
    - PatchDiscriminator:  70x70 PatchGAN discriminator
    - Pix2PixLoss:        Combined adversarial + L1 loss
"""

import torch
import torch.nn as nn


# U-Net Generator

class UNetDown(nn.Module):
    """Encoder block: Conv → [BatchNorm] → LeakyReLU"""

    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUp(nn.Module):
    """Decoder block: ConvTranspose → BatchNorm → [Dropout] → ReLU + skip concat"""

    def __init__(self, in_ch, out_ch, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    """
    U-Net generator with skip connections.

    Input:  (B, 1, 512, 512)  low-dose CT
    Output: (B, 1, 512, 512)  predicted full-dose CT
    """

    def __init__(self, in_ch=1, out_ch=1, base_filters=64):
        super().__init__()
        bf = base_filters

        # Encoder (no BN on first layer)
        self.down1 = UNetDown(in_ch, bf, use_bn=False)    # 512 --> 256
        self.down2 = UNetDown(bf, bf * 2)        # 256 --> 128
        self.down3 = UNetDown(bf * 2, bf * 4)    # 128 --> 64
        self.down4 = UNetDown(bf * 4, bf * 8)    # 64  --> 32
        self.down5 = UNetDown(bf * 8, bf * 8)    # 32 --> 16
        self.down6 = UNetDown(bf * 8, bf * 8)    # 16  --> 8
        self.down7 = UNetDown(bf * 8, bf * 8)   # 8 --> 4
        self.down8 = UNetDown(bf * 8, bf * 8, use_bn=False)  # 4 → 2

        # Decoder (with skip connections, dropout on first 3)
        self.up1 = UNetUp(bf * 8, bf * 8, use_dropout=True)      # 2  --> 4
        self.up2 = UNetUp(bf * 8 * 2, bf * 8, use_dropout=True)  # 4 --> 8
        self.up3 = UNetUp(bf * 8 * 2, bf * 8, use_dropout=True)  # 8  --> 16
        self.up4 = UNetUp(bf * 8 * 2, bf * 8)                    # 16--> 32
        self.up5 = UNetUp(bf * 8 * 2, bf * 4)                    # 32 -->64
        self.up6 = UNetUp(bf * 4 * 2, bf * 2)                    # 64 --> 128
        self.up7 = UNetUp(bf * 2 * 2, bf)                        # 128 --> 256

        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(bf * 2, out_ch, 4, stride=2, padding=1),  # 256 --> 512
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


# PatchGAN Discriminator

class PatchDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator.

    Takes concatenated (input, target) or (input, generated) pair.
    Outputs a grid of real/fake predictions.

    Input:  (B, 2, 512, 512)  — concatenated low-dose + full-dose/generated
    Output: (B, 1, 30, 30)    — patch-level real/fake predictions
    """

    def __init__(self, in_ch=2, base_filters=64):
        super().__init__()
        bf = base_filters

        self.model = nn.Sequential(
            # Layer 1 — no BatchNorm
            nn.Conv2d(in_ch, bf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Conv2d(bf, bf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv2d(bf * 2, bf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4 — stride 1
            nn.Conv2d(bf * 4, bf * 8, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(bf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output — single channel prediction map
            nn.Conv2d(bf * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, input_img, target_img):
        x = torch.cat([input_img, target_img], dim=1)
        return self.model(x)


# Loss

class Pix2PixLoss:
    """
    Combined pix2pix loss.

    Generator:     L_adv (fool discriminator) + lambda_l1 * L1 (pixel accuracy)
    Discriminator: standard GAN loss on real + fake pairs
    """

    def __init__(self, lambda_l1=100.0, device="cuda"):
        self.lambda_l1 = lambda_l1
        self.bce = nn.BCEWithLogitsLoss().to(device)
        self.l1 = nn.L1Loss().to(device)

    def generator_loss(self, disc_fake_output, generated, target):
        """Generator wants discriminator to think fake is real."""
        adv_loss = self.bce(disc_fake_output, torch.ones_like(disc_fake_output))
        l1_loss = self.l1(generated, target)
        return adv_loss + self.lambda_l1 * l1_loss, adv_loss, l1_loss

    def discriminator_loss(self, disc_real_output, disc_fake_output):
        """Discriminator wants to correctly classify real vs fake."""
        real_loss = self.bce(disc_real_output, torch.ones_like(disc_real_output))
        fake_loss = self.bce(disc_fake_output, torch.zeros_like(disc_fake_output))
        return (real_loss + fake_loss) * 0.5


# Quick smoke check

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen = UNetGenerator().to(device)
    disc = PatchDiscriminator().to(device)

    # Count parameters
    gen_params = sum(p.numel() for p in gen.parameters()) / 1e6
    disc_params = sum(p.numel() for p in disc.parameters()) / 1e6
    print(f"Generator:     {gen_params:.1f}M parameters")
    print(f"Discriminator: {disc_params:.1f}M parameters")

    # Test forward pass
    low_dose = torch.randn(1, 1, 512, 512).to(device)
    full_dose = torch.randn(1, 1, 512, 512).to(device)

    fake = gen(low_dose)
    print(f"\nGenerator:     {low_dose.shape} → {fake.shape}")

    pred = disc(low_dose, fake)
    print(f"Discriminator: {low_dose.shape} + {fake.shape} → {pred.shape}")

    print("\nPix2Pix model OK!")