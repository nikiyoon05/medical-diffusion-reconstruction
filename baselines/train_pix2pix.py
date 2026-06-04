"""
train_pix2pix.py

!!! Since this is the baseline comparision, most of this is basically just
copied from the implementation of baseline:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/models


Train the pix2pix baseline on Modal.

Run:  modal run baselines/train_pix2pix.py
Get results: modal volume get ldct-data /results/pix2pix ./results_pix2pix

Trains for ~50 epochs on the paired low-dose → full-dose CT data.
Saves checkpoints, sample images, and metrics to the Modal volume.
"""

import modal
import os

app = modal.App("ldct-pix2pix")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
    )
)

vol = modal.Volume.from_name("ldct-data")


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="A10G",
    timeout=7200,  # 2 hours
)
def train():
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    import json
    import sys

    # --- Add project root to path so we can import our modules ---
    sys.path.insert(0, "/data/project")

    # --- Import  modules ---

    # INLINE: Dataset (same as data/dataset.py)
    from torch.utils.data import Dataset

    class LDCTDataset(Dataset):
        def __init__(self, root_dir, split="train", img_size=512):
            self.root_dir = root_dir
            self.img_size = img_size
            self.pairs = []

            all_patients = sorted([
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ])

            n = len(all_patients)
            train_end = int(0.70 * n)
            val_end = int(0.85 * n)

            if split == "train":
                patients = all_patients[:train_end]
            elif split == "val":
                patients = all_patients[train_end:val_end]
            elif split == "test":
                patients = all_patients[val_end:]
            else:
                raise ValueError(f"Unknown split: {split}")

            for patient in patients:
                ld_dir = os.path.join(root_dir, patient, "low_dose")
                fd_dir = os.path.join(root_dir, patient, "full_dose")
                if not os.path.isdir(ld_dir) or not os.path.isdir(fd_dir):
                    continue

                ld_files = sorted([f for f in os.listdir(ld_dir) if f.endswith(".npy")])
                fd_files = sorted([f for f in os.listdir(fd_dir) if f.endswith(".npy")])
                n_slices = min(len(ld_files), len(fd_files))

                for i in range(n_slices):
                    self.pairs.append((
                        os.path.join(ld_dir, ld_files[i]),
                        os.path.join(fd_dir, fd_files[i]),
                    ))

            print(f"LDCTDataset [{split}]: {len(patients)} patients, {len(self.pairs)} slices")

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            ld_path, fd_path = self.pairs[idx]
            low_dose = np.load(ld_path)
            full_dose = np.load(fd_path)

            low_dose = torch.from_numpy(low_dose).unsqueeze(0)
            full_dose = torch.from_numpy(full_dose).unsqueeze(0)

            if self.img_size != 512:
                low_dose = torch.nn.functional.interpolate(
                    low_dose.unsqueeze(0), size=self.img_size, mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                full_dose = torch.nn.functional.interpolate(
                    full_dose.unsqueeze(0), size=self.img_size, mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            low_dose = low_dose * 2 - 1
            full_dose = full_dose * 2 - 1

            return {"low_dose": low_dose, "full_dose": full_dose}

    # INLINE: Pix2Pix model (same as baselines/pix2pix.py)

    class UNetDown(nn.Module):
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
        def __init__(self, in_ch=1, out_ch=1, base_filters=64):
            super().__init__()
            bf = base_filters
            self.down1 = UNetDown(in_ch, bf, use_bn=False)
            self.down2 = UNetDown(bf, bf * 2)
            self.down3 = UNetDown(bf * 2, bf * 4)
            self.down4 = UNetDown(bf * 4, bf * 8)
            self.down5 = UNetDown(bf * 8, bf * 8)
            self.down6 = UNetDown(bf * 8, bf * 8)
            self.down7 = UNetDown(bf * 8, bf * 8)
            self.down8 = UNetDown(bf * 8, bf * 8, use_bn=False)

            self.up1 = UNetUp(bf * 8, bf * 8, use_dropout=True)
            self.up2 = UNetUp(bf * 8 * 2, bf * 8, use_dropout=True)
            self.up3 = UNetUp(bf * 8 * 2, bf * 8, use_dropout=True)
            self.up4 = UNetUp(bf * 8 * 2, bf * 8)
            self.up5 = UNetUp(bf * 8 * 2, bf * 4)
            self.up6 = UNetUp(bf * 4 * 2, bf * 2)
            self.up7 = UNetUp(bf * 2 * 2, bf)

            self.final = nn.Sequential(
                nn.ConvTranspose2d(bf * 2, out_ch, 4, stride=2, padding=1),
                nn.Tanh(),
            )

        def forward(self, x):
            d1 = self.down1(x)
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            d4 = self.down4(d3)
            d5 = self.down5(d4)
            d6 = self.down6(d5)
            d7 = self.down7(d6)
            d8 = self.down8(d7)

            u1 = self.up1(d8, d7)
            u2 = self.up2(u1, d6)
            u3 = self.up3(u2, d5)
            u4 = self.up4(u3, d4)
            u5 = self.up5(u4, d3)
            u6 = self.up6(u5, d2)
            u7 = self.up7(u6, d1)
            return self.final(u7)

    class PatchDiscriminator(nn.Module):
        def __init__(self, in_ch=2, base_filters=64):
            super().__init__()
            bf = base_filters
            self.model = nn.Sequential(
                nn.Conv2d(in_ch, bf, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(bf, bf * 2, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(bf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(bf * 2, bf * 4, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(bf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(bf * 4, bf * 8, 4, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(bf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(bf * 8, 1, 4, stride=1, padding=1),
            )

        def forward(self, input_img, target_img):
            x = torch.cat([input_img, target_img], dim=1)
            return self.model(x)

    # Config
    DATA_DIR = "/data/processed"
    RESULTS_DIR = "/data/results/pix2pix"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/samples", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/checkpoints", exist_ok=True)

    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    LR = 2e-4
    BETA1 = 0.5
    LAMBDA_L1 = 100.0
    IMG_SIZE = 256  # Train at 256 for speed; pix2pix doesn't need 512

    device = torch.device("cuda")

    # Data
    print("Loading data...")
    train_ds = LDCTDataset(DATA_DIR, split="train", img_size=IMG_SIZE)
    val_ds = LDCTDataset(DATA_DIR, split="val", img_size=IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Model adn optimizers
    print("Initializing models...")
    gen = UNetGenerator(in_ch=1, out_ch=1).to(device)
    disc = PatchDiscriminator(in_ch=2).to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=LR, betas=(BETA1, 0.999))

    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    gen_params = sum(p.numel() for p in gen.parameters()) / 1e6
    disc_params = sum(p.numel() for p in disc.parameters()) / 1e6
    print(f"Generator:     {gen_params:.1f}M params")
    print(f"Discriminator: {disc_params:.1f}M params")
    print(f"Training: {len(train_ds)} slices, {NUM_EPOCHS} epochs, "
          f"batch_size={BATCH_SIZE}, img_size={IMG_SIZE}")

    # Metrics
    def compute_psnr(pred, target):
        """PSNR in dB. Inputs in [-1, 1]."""
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float("inf")
        # Data range is 2.0 (from -1 to 1)
        return 10 * torch.log10(4.0 / mse).item()

    def compute_ssim(pred, target):
        """Simplified SSIM for single-channel images. Inputs in [-1, 1]."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_p = torch.mean(pred)
        mu_t = torch.mean(target)
        sigma_p = torch.var(pred)
        sigma_t = torch.var(target)
        sigma_pt = torch.mean((pred - mu_p) * (target - mu_t))

        num = (2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)
        den = (mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p + sigma_t + C2)
        return (num / den).item()

    # Training loop
    history = {"epoch": [], "g_loss": [], "d_loss": [], "val_psnr": [], "val_ssim": []}

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        gen.train()
        disc.train()
        epoch_g_loss = 0
        epoch_d_loss = 0

        for batch in train_loader:
            low = batch["low_dose"].to(device)
            full = batch["full_dose"].to(device)

            # --- Train Discriminator ---
            fake = gen(low).detach()

            pred_real = disc(low, full)
            pred_fake = disc(low, fake)

            loss_d_real = bce_loss(pred_real, torch.ones_like(pred_real))
            loss_d_fake = bce_loss(pred_fake, torch.zeros_like(pred_fake))
            loss_d = (loss_d_real + loss_d_fake) * 0.5

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # --- Train Generator ---
            fake = gen(low)
            pred_fake = disc(low, fake)

            loss_g_adv = bce_loss(pred_fake, torch.ones_like(pred_fake))
            loss_g_l1 = l1_loss(fake, full)
            loss_g = loss_g_adv + LAMBDA_L1 * loss_g_l1

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()

        avg_g = epoch_g_loss / len(train_loader)
        avg_d = epoch_d_loss / len(train_loader)

        # --- Validation metrics ---
        gen.eval()
        val_psnr = 0
        val_ssim = 0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                low = batch["low_dose"].to(device)
                full = batch["full_dose"].to(device)
                fake = gen(low)

                for i in range(fake.size(0)):
                    val_psnr += compute_psnr(fake[i], full[i])
                    val_ssim += compute_ssim(fake[i], full[i])
                    n_val += 1

        val_psnr /= max(n_val, 1)
        val_ssim /= max(n_val, 1)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"G: {avg_g:.4f}  D: {avg_d:.4f} | "
              f"Val PSNR: {val_psnr:.2f} dB  SSIM: {val_ssim:.4f}")

        history["epoch"].append(epoch)
        history["g_loss"].append(avg_g)
        history["d_loss"].append(avg_d)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        # --- Save sample images every 10 epochs ---
        if epoch % 10 == 0 or epoch == 1:
            gen.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                low = sample_batch["low_dose"][:4].to(device)
                full = sample_batch["full_dose"][:4].to(device)
                fake = gen(low)

            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            for i in range(min(4, low.size(0))):
                axes[0, i].imshow(low[i, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
                axes[0, i].set_title("Low-dose", fontsize=10)
                axes[0, i].axis("off")

                axes[1, i].imshow(fake[i, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
                axes[1, i].set_title("Pix2Pix output", fontsize=10)
                axes[1, i].axis("off")

                axes[2, i].imshow(full[i, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
                axes[2, i].set_title("Full-dose (GT)", fontsize=10)
                axes[2, i].axis("off")

            plt.suptitle(f"Epoch {epoch}", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{RESULTS_DIR}/samples/epoch_{epoch:03d}.png", dpi=120)
            plt.close()

        # --- Save checkpoint every 25 epochs ---
        if epoch % 25 == 0 or epoch == NUM_EPOCHS:
            torch.save({
                "epoch": epoch,
                "gen_state": gen.state_dict(),
                "disc_state": disc.state_dict(),
                "opt_g_state": opt_g.state_dict(),
                "opt_d_state": opt_d.state_dict(),
                "history": history,
            }, f"{RESULTS_DIR}/checkpoints/epoch_{epoch:03d}.pt")

    # ---------------Save final results-----------------

    # Loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["epoch"], history["g_loss"], label="Generator")
    ax1.plot(history["epoch"], history["d_loss"], label="Discriminator")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()

    ax2.plot(history["epoch"], history["val_psnr"], label="PSNR (dB)", color="tab:blue")
    ax2_r = ax2.twinx()
    ax2_r.plot(history["epoch"], history["val_ssim"], label="SSIM", color="tab:orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("PSNR (dB)", color="tab:blue")
    ax2_r.set_ylabel("SSIM", color="tab:orange")
    ax2.set_title("Validation Metrics")

    plt.suptitle("Pix2Pix Baseline Results", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/training_curves.png", dpi=120)
    plt.close()

    # Save metrics as JSON
    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    vol.commit()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final Val PSNR: {history['val_psnr'][-1]:.2f} dB")
    print(f"Final Val SSIM: {history['val_ssim'][-1]:.4f}")
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("Download with:")
    print("  modal volume get ldct-data /results/pix2pix ./results_pix2pix")


@app.local_entrypoint()
def main():
    train.remote()