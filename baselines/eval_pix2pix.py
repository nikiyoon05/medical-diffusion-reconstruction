"""
eval_pix2pix.py

Evaluate the trained pix2pix baseline on the test set.
*** Most of this just copied from data/train_pix2pix.py, but now
just for evaluation.

Generates:
    -Per-slice PSNR and SSIM
    - Side-by-side comparison images (low-dose | pix2pix | full-dose)
    - Residual maps showing what the model changed
    - Summary metrics

Run:  modal run baselines/eval_pix2pix.py
Get results: modal volume get ldct-data /results/pix2pix/eval ./eval_pix2pix
Ideally only need to run this once. This has been run and can see results
on modal storage.
"""

import modal
import os

app = modal.App("ldct-pix2pix-eval")

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
    timeout=1800,
)
def evaluate():
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import json

    # ================================================================
    # INLINE: Dataset
    # ================================================================
    from torch.utils.data import Dataset, DataLoader

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

            self.patients = patients

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
                        patient,
                        i,
                    ))

            print(f"LDCTDataset [{split}]: {len(patients)} patients ({patients}), "
                  f"{len(self.pairs)} slices")

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            ld_path, fd_path, patient, slice_idx = self.pairs[idx]
            low_dose = np.load(ld_path)
            full_dose = np.load(fd_path)

            low_dose = torch.from_numpy(low_dose).unsqueeze(0)
            full_dose = torch.from_numpy(full_dose).unsqueeze(0)

            if self.img_size != 512:
                low_dose = nn.functional.interpolate(
                    low_dose.unsqueeze(0), size=self.img_size, mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                full_dose = nn.functional.interpolate(
                    full_dose.unsqueeze(0), size=self.img_size, mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            low_dose = low_dose * 2 - 1
            full_dose = full_dose * 2 - 1

            return {
                "low_dose": low_dose,
                "full_dose": full_dose,
                "patient": patient,
                "slice_idx": slice_idx,
            }

    # IGenerator (must match training architecture exactly)

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

    # metrics

    def compute_psnr(pred, target):
        """PSNR in dB. Inputs in [-1, 1], so data range = 2."""
        mse = torch.mean((pred - target) ** 2).item()
        if mse == 0:
            return float("inf")
        return 10 * np.log10(4.0 / mse)

    def compute_ssim(pred, target, window_size=11):
        """
        SSIM for single-channel images. Inputs in [-1, 1].
        Uses a Gaussian sliding window for proper per-patch computation.
        """
        C1 = (0.01 * 2) ** 2  # data range = 2
        C2 = (0.03 * 2) ** 2

        # Create Gaussian kernel
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(0) * g.unsqueeze(1)  # 2D kernel
        window = window.unsqueeze(0).unsqueeze(0).to(pred.device)  # (1, 1, k, k)

        mu1 = nn.functional.conv2d(pred, window, padding=window_size // 2)
        mu2 = nn.functional.conv2d(target, window, padding=window_size // 2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = nn.functional.conv2d(pred ** 2, window, padding=window_size // 2) - mu1_sq
        sigma2_sq = nn.functional.conv2d(target ** 2, window, padding=window_size // 2) - mu2_sq
        sigma12 = nn.functional.conv2d(pred * target, window, padding=window_size // 2) - mu12

        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean().item()

    # Config
    DATA_DIR = "/data/processed"
    RESULTS_DIR = "/data/results/pix2pix_v2"
    EVAL_DIR = f"{RESULTS_DIR}/eval"
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(f"{EVAL_DIR}/comparisons", exist_ok=True)

    IMG_SIZE = 256
    CHECKPOINT = f"{RESULTS_DIR}/checkpoints/epoch_050.pt"

    device = torch.device("cuda")

    # Load model
    print("Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    print(f"  Loaded epoch {checkpoint['epoch']}")

    gen = UNetGenerator(in_ch=1, out_ch=1).to(device)
    gen.load_state_dict(checkpoint["gen_state"])
    gen.eval()

    # Print training history summary
    hist = checkpoint.get("history", {})
    if hist:
        print(f"  Final train G loss: {hist['g_loss'][-1]:.4f}")
        print(f"  Final val PSNR: {hist['val_psnr'][-1]:.2f} dB")
        print(f"  Final val SSIM: {hist['val_ssim'][-1]:.4f}")

    #--------------Test set evaluation-----------
    print("\nLoading test set...")
    test_ds = LDCTDataset(DATA_DIR, split="test", img_size=IMG_SIZE)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Also compute metrics on low-dose input for reference
    # (how much did pix2pix actually improve over doing nothing?)
    results = []
    all_psnr_model = []
    all_psnr_input = []
    all_ssim_model = []
    all_ssim_input = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            low = batch["low_dose"].to(device)
            full = batch["full_dose"].to(device)
            patients = batch["patient"]
            slice_idxs = batch["slice_idx"]

            fake = gen(low)

            for i in range(low.size(0)):
                psnr_model = compute_psnr(fake[i:i+1], full[i:i+1])
                ssim_model = compute_ssim(fake[i:i+1], full[i:i+1])
                psnr_input = compute_psnr(low[i:i+1], full[i:i+1])
                ssim_input = compute_ssim(low[i:i+1], full[i:i+1])

                all_psnr_model.append(psnr_model)
                all_ssim_model.append(ssim_model)
                all_psnr_input.append(psnr_input)
                all_ssim_input.append(ssim_input)

                results.append({
                    "patient": patients[i],
                    "slice": int(slice_idxs[i]),
                    "psnr_model": round(psnr_model, 2),
                    "ssim_model": round(ssim_model, 4),
                    "psnr_input": round(psnr_input, 2),
                    "ssim_input": round(ssim_input, 4),
                })

    #--------------summary-----------------------
    mean_psnr_model = np.mean(all_psnr_model)
    mean_ssim_model = np.mean(all_ssim_model)
    mean_psnr_input = np.mean(all_psnr_input)
    mean_ssim_input = np.mean(all_ssim_input)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"  {'':20s} {'PSNR (dB)':>10s}  {'SSIM':>8s}")
    print(f"  {'Low-dose (no model)':20s} {mean_psnr_input:10.2f}  {mean_ssim_input:8.4f}")
    print(f"  {'Pix2Pix':20s} {mean_psnr_model:10.2f}  {mean_ssim_model:8.4f}")
    print(f"  {'Improvement':20s} "
          f"{mean_psnr_model - mean_psnr_input:+10.2f}  "
          f"{mean_ssim_model - mean_ssim_input:+8.4f}")
    print(f"\n  Test slices: {len(results)}")

    # Per-patient breakdown
    print("\n  Per-patient breakdown:")
    for patient in sorted(set(r["patient"] for r in results)):
        pr = [r for r in results if r["patient"] == patient]
        p_psnr = np.mean([r["psnr_model"] for r in pr])
        p_ssim = np.mean([r["ssim_model"] for r in pr])
        print(f"    {patient}: PSNR={p_psnr:.2f} dB, SSIM={p_ssim:.4f} ({len(pr)} slices)")

    #--------------Generate comparison images--------------
    print("\nGenerating comparison images...")
    test_loader_single = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Save comparisons for a few representative slices per patient
    saved_per_patient = {}
    MAX_PER_PATIENT = 5 # with a val set of 3 patients gives us 15 images

    with torch.no_grad():
        for idx, batch in enumerate(test_loader_single):
            patient = batch["patient"][0]
            slice_idx = int(batch["slice_idx"][0])

            if saved_per_patient.get(patient, 0) >= MAX_PER_PATIENT:
                continue

            low = batch["low_dose"].to(device)
            full = batch["full_dose"].to(device)
            fake = gen(low)

            # Compute per image metrics for the title
            psnr = compute_psnr(fake, full)
            ssim = compute_ssim(fake, full)

            # Convert to numpy for plotting
            low_np = low[0, 0].cpu().numpy()
            fake_np = fake[0, 0].cpu().numpy()
            full_np = full[0, 0].cpu().numpy()
            residual = np.abs(fake_np - full_np)

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(low_np, cmap="gray", vmin=-1, vmax=1)
            axes[0].set_title("Low-dose input", fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(fake_np, cmap="gray", vmin=-1, vmax=1)
            axes[1].set_title(f"Pix2Pix output\nPSNR={psnr:.1f} SSIM={ssim:.3f}", fontsize=12)
            axes[1].axis("off")

            axes[2].imshow(full_np, cmap="gray", vmin=-1, vmax=1)
            axes[2].set_title("Full-dose (ground truth)", fontsize=12)
            axes[2].axis("off")

            axes[3].imshow(residual, cmap="hot", vmin=0, vmax=0.5)
            axes[3].set_title("Residual (error)", fontsize=12)
            axes[3].axis("off")

            plt.suptitle(f"{patient} / slice {slice_idx}", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{EVAL_DIR}/comparisons/{patient}_slice{slice_idx:03d}.png",
                        dpi=120, bbox_inches="tight")
            plt.close()

            saved_per_patient[patient] = saved_per_patient.get(patient, 0) + 1

    # Save metrics
    summary = {
        "model": "pix2pix",
        "checkpoint": CHECKPOINT,
        "test_slices": len(results),
        "mean_psnr_model": round(mean_psnr_model, 2),
        "mean_ssim_model": round(mean_ssim_model, 4),
        "mean_psnr_input": round(mean_psnr_input, 2),
        "mean_ssim_input": round(mean_ssim_input, 4),
        "psnr_improvement": round(mean_psnr_model - mean_psnr_input, 2),
        "ssim_improvement": round(mean_ssim_model - mean_ssim_input, 4),
        "per_slice": results,
    }

    with open(f"{EVAL_DIR}/test_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    vol.commit()

    print(f"\nResults saved to: {EVAL_DIR}")
    print("Download with:")
    print("  modal volume get ldct-data /results/pix2pix/eval ./eval_pix2pix")


@app.local_entrypoint()
def main():
    evaluate.remote()