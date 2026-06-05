"""
eval_val.py

Evaluation on the VALIDATION set with midpoint sampling.

This file has some redundancy with testing, but I made it to easily compare
baseline versus rectified flow models on the validation set (15% of patients), 
so the test set remains untouched in case we change the model again.

Configure the flags below before running:
    GENERATE_SAMPLES  — save comparison images (low-dose | diffusion | GT | residual)
    COMPARE_PIX2PIX  — add pix2pix column to comparison images
    MAX_SAMPLES   — how many comparison images to save

Run: modal run eval_val.py
Get: modal volume get ldct-data /results/diffusion_v2/validation_samples ./validation_samples
"""

import modal

app = modal.App("ldct-eval-val")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "numpy", "matplotlib", "medvae")
    .add_local_python_source("data")
    .add_local_python_source("model")
)

vol = modal.Volume.from_name("ldct-data")

# =-----CONFIG FLAGS-----
CHECKPOINT = "/data/results/diffusion_v2/checkpoints/best.pt"
PIX2PIX_CHECKPOINT = "/data/results/pix2pix/checkpoints/epoch_050.pt"
SAMPLE_STEPS = 100
GENERATE_SAMPLES = True # Save comparison images
COMPARE_PIX2PIX = True # Include pix2pix in comparisons
MAX_SAMPLES = 12    # Number of comparison images to save


@app.function(image=image, volumes={"/data": vol}, gpu="A100", timeout=3600)
def eval_val():
    import torch
    import torch.nn as nn
    import numpy as np
    import os

    from torch.utils.data import DataLoader
    from data.latent_dataset import LDCTLatentDataset
    from data.dataset import LDCTDataset
    from model.rectified_flow import RectifiedFlow
    from model.unet import ConditionalUNet
    from model.vae import MedVAEWrapper

    device = torch.device("cuda")

    RESULTS_DIR = "/data/results/diffusion_v2/validation_samples"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -----Load diffusion model (EMA weights)--------
    print("Loading diffusion checkpoint...")
    ckpt = torch.load(CHECKPOINT, map_location=device)
    print(f"  Epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.5f}")

    backbone = ConditionalUNet(
        in_channels=4, cond_channels=4, base_channels=128,
        channel_mults=(1, 2, 4, 4), num_res_blocks=2,
        dropout=0.0, attn_resolutions=(8,),
    ).to(device)
    backbone.load_state_dict(ckpt["backbone_state"])
    backbone.eval()

    flow = RectifiedFlow(backbone).to(device)
    vae = MedVAEWrapper(model_name="medvae_8_4_2d", device=device)

    # -----Load pix2pix generator (if comparing)-------
    pix2pix_gen = None
    if COMPARE_PIX2PIX and os.path.exists(PIX2PIX_CHECKPOINT):
        print("Loading pix2pix checkpoint...")

        # Inline pix2pix generator (must match training architecture)
        class UNetDown(nn.Module):
            def __init__(self, in_ch, out_ch, use_bn=True):
                super().__init__()
                layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)]
                if use_bn: layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                self.block = nn.Sequential(*layers)
            def forward(self, x): return self.block(x)

        class UNetUp(nn.Module):
            def __init__(self, in_ch, out_ch, use_dropout=False):
                super().__init__()
                layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
                          nn.BatchNorm2d(out_ch)]
                if use_dropout: layers.append(nn.Dropout(0.5))
                layers.append(nn.ReLU(inplace=True))
                self.block = nn.Sequential(*layers)
            def forward(self, x, skip):
                return torch.cat([self.block(x), skip], dim=1)

        class UNetGenerator(nn.Module):
            def __init__(self, in_ch=1, out_ch=1, bf=64):
                super().__init__()
                self.down1 = UNetDown(in_ch, bf, use_bn=False)
                self.down2 = UNetDown(bf, bf*2)
                self.down3 = UNetDown(bf*2, bf*4)
                self.down4 = UNetDown(bf*4, bf*8)
                self.down5 = UNetDown(bf*8, bf*8)
                self.down6 = UNetDown(bf*8, bf*8)
                self.down7 = UNetDown(bf*8, bf*8)
                self.down8 = UNetDown(bf*8, bf*8, use_bn=False)
                self.up1 = UNetUp(bf*8, bf*8, use_dropout=True)
                self.up2 = UNetUp(bf*8*2, bf*8, use_dropout=True)
                self.up3 = UNetUp(bf*8*2, bf*8, use_dropout=True)
                self.up4 = UNetUp(bf*8*2, bf*8)
                self.up5 = UNetUp(bf*8*2, bf*4)
                self.up6 = UNetUp(bf*4*2, bf*2)
                self.up7 = UNetUp(bf*2*2, bf)
                self.final = nn.Sequential(
                    nn.ConvTranspose2d(bf*2, out_ch, 4, stride=2, padding=1), nn.Tanh())

            def forward(self, x):
                d1=self.down1(x); d2=self.down2(d1); d3=self.down3(d2); d4=self.down4(d3)
                d5=self.down5(d4); d6=self.down6(d5); d7=self.down7(d6); d8=self.down8(d7)
                u1=self.up1(d8,d7); u2=self.up2(u1,d6); u3=self.up3(u2,d5); u4=self.up4(u3,d4)
                u5=self.up5(u4,d3); u6=self.up6(u5,d2); u7=self.up7(u6,d1)
                return self.final(u7)

        pix2pix_gen = UNetGenerator().to(device)
        p2p_ckpt = torch.load(PIX2PIX_CHECKPOINT, map_location=device)
        pix2pix_gen.load_state_dict(p2p_ckpt["gen_state"])
        pix2pix_gen.eval()
        print("  Pix2pix loaded")
    elif COMPARE_PIX2PIX:
        print(f"  WARNING: pix2pix checkpoint not found at {PIX2PIX_CHECKPOINT}")
        print("  Running without pix2pix comparison")

    # Datasets
    val_latent = LDCTLatentDataset("/data/latents", split="val", augment=False)
    val_loader = DataLoader(val_latent, batch_size=4, shuffle=False)

    # For pix2pix: need raw images at 256x256
    val_images = None
    if pix2pix_gen is not None:
        val_images = LDCTDataset("/data/processed", split="val", img_size=256)

    # ------Metrics------
    def compute_psnr(pred, target):
        mse = torch.mean((pred - target) ** 2).item()
        return float("inf") if mse == 0 else 10 * np.log10(4.0 / mse)

    def compute_ssim(pred, target, window_size=11):
        C1 = (0.01 * 2) ** 2
        C2 = (0.03 * 2) ** 2
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0).to(pred.device)
        mu1 = nn.functional.conv2d(pred, window, padding=window_size // 2)
        mu2 = nn.functional.conv2d(target, window, padding=window_size // 2)
        mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        s1 = nn.functional.conv2d(pred ** 2, window, padding=window_size // 2) - mu1_sq
        s2 = nn.functional.conv2d(target ** 2, window, padding=window_size // 2) - mu2_sq
        s12 = nn.functional.conv2d(pred * target, window, padding=window_size // 2) - mu12
        ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
        return ssim_map.mean().item()

    # ---------------Run evaluation------------------------------------------
    print(f"\nEval on {len(val_latent)} val slices, {SAMPLE_STEPS} midpoint steps...")

    psnr_diff, ssim_diff = [], []
    psnr_input, ssim_input = [], []
    psnr_p2p, ssim_p2p = [], []

    n_done = 0
    sample_idx = 0  # tracks position for image saving

    with torch.no_grad():
        for batch_i, batch in enumerate(val_loader):
            z_cond = batch["z_low"].to(device)
            z_target = batch["z_full"].to(device)
            B = z_cond.size(0)

            # Diffusion output
            z_pred = flow.sample_midpoint(z_cond, num_steps=SAMPLE_STEPS, device=device)

            pred_imgs = vae.decode(z_pred)   # (B, 1, 512, 512)
            target_imgs = vae.decode(z_target)    # (B, 1, 512, 512)
            input_imgs = vae.decode(z_cond)   # (B, 1, 512, 512)

            # Pix2pix output (if enabled)
            p2p_imgs = None
            if pix2pix_gen is not None and val_images is not None:
                p2p_list = []
                for j in range(B):
                    img_idx = n_done + j
                    if img_idx < len(val_images):
                        img_batch = val_images[img_idx]
                        low_256 = img_batch["low_dose"].unsqueeze(0).to(device)
                        p2p_out = pix2pix_gen(low_256)
                        # Resize 256→512 for fair visual comparison
                        p2p_out = nn.functional.interpolate(
                            p2p_out, size=512, mode="bilinear", align_corners=False)
                        p2p_list.append(p2p_out)
                if p2p_list:
                    p2p_imgs = torch.cat(p2p_list, dim=0)

            # Compute metrics per slice
            for i in range(B):
                psnr_diff.append(compute_psnr(pred_imgs[i:i+1], target_imgs[i:i+1]))
                ssim_diff.append(compute_ssim(pred_imgs[i:i+1], target_imgs[i:i+1]))
                psnr_input.append(compute_psnr(input_imgs[i:i+1], target_imgs[i:i+1]))
                ssim_input.append(compute_ssim(input_imgs[i:i+1], target_imgs[i:i+1]))

                if p2p_imgs is not None and i < p2p_imgs.size(0):
                    psnr_p2p.append(compute_psnr(p2p_imgs[i:i+1], target_imgs[i:i+1]))
                    ssim_p2p.append(compute_ssim(p2p_imgs[i:i+1], target_imgs[i:i+1]))

            # Save comparison images
            if GENERATE_SAMPLES and sample_idx < MAX_SAMPLES:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                for i in range(B):
                    if sample_idx >= MAX_SAMPLES:
                        break
                    # Only save every Nth slice to get variety
                    if (n_done + i) % (len(val_latent) // MAX_SAMPLES) != 0:
                        continue

                    inp = input_imgs[i, 0].cpu().numpy()
                    diff = pred_imgs[i, 0].cpu().numpy()
                    gt = target_imgs[i, 0].cpu().numpy()
                    resid = np.abs(diff - gt)

                    if p2p_imgs is not None and i < p2p_imgs.size(0):
                        # 5-column: input | pix2pix | diffusion | GT | residual
                        p2p = p2p_imgs[i, 0].cpu().numpy()
                        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
                        axes[0].imshow(inp, cmap="gray", vmin=-1, vmax=1)
                        axes[0].set_title("Low-dose input"); axes[0].axis("off")
                        axes[1].imshow(p2p, cmap="gray", vmin=-1, vmax=1)
                        p2p_psnr = compute_psnr(p2p_imgs[i:i+1], target_imgs[i:i+1])
                        axes[1].set_title(f"Pix2Pix\nPSNR={p2p_psnr:.1f}"); axes[1].axis("off")
                        axes[2].imshow(diff, cmap="gray", vmin=-1, vmax=1)
                        d_psnr = compute_psnr(pred_imgs[i:i+1], target_imgs[i:i+1])
                        axes[2].set_title(f"Diffusion\nPSNR={d_psnr:.1f}"); axes[2].axis("off")
                        axes[3].imshow(gt, cmap="gray", vmin=-1, vmax=1)
                        axes[3].set_title("Full-dose (GT)"); axes[3].axis("off")
                        axes[4].imshow(resid, cmap="hot", vmin=0, vmax=0.5)
                        axes[4].set_title("Residual (diff)"); axes[4].axis("off")
                    else:
                        # 4-column: input | diffusion | GT | residual
                        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                        axes[0].imshow(inp, cmap="gray", vmin=-1, vmax=1)
                        axes[0].set_title("Low-dose input"); axes[0].axis("off")
                        axes[1].imshow(diff, cmap="gray", vmin=-1, vmax=1)
                        d_psnr = compute_psnr(pred_imgs[i:i+1], target_imgs[i:i+1])
                        axes[1].set_title(f"Diffusion\nPSNR={d_psnr:.1f}"); axes[1].axis("off")
                        axes[2].imshow(gt, cmap="gray", vmin=-1, vmax=1)
                        axes[2].set_title("Full-dose (GT)"); axes[2].axis("off")
                        axes[3].imshow(resid, cmap="hot", vmin=0, vmax=0.5)
                        axes[3].set_title("Residual"); axes[3].axis("off")

                    plt.tight_layout()
                    plt.savefig(f"{RESULTS_DIR}/val_{sample_idx:03d}.png",
                                dpi=150, bbox_inches="tight")
                    plt.close()
                    sample_idx += 1

            n_done += B
            if n_done % 100 == 0:
                print(f"  {n_done}/{len(val_latent)} — "
                      f"PSNR: {np.mean(psnr_diff):.2f}, "
                      f"SSIM: {np.mean(ssim_diff):.4f}")

    # ---------------------------Summary-------------------
    mp = np.mean(psnr_diff); ms = np.mean(ssim_diff)
    ip = np.mean(psnr_input); iss_ = np.mean(ssim_input)

    print("\n" + "=" * 60)
    print(f"VALIDATION RESULTS (midpoint, {SAMPLE_STEPS} steps)")
    print("=" * 60)
    print(f"{'':22s} {'PSNR (dB)':>10s}  {'SSIM':>8s}")
    print(f"{'Low-dose (no model)':22s} {ip:10.2f}  {iss_:8.4f}")

    if psnr_p2p:
        pp = np.mean(psnr_p2p); ps = np.mean(ssim_p2p)
        print(f"{'Pix2Pix':22s} {pp:10.2f}  {ps:8.4f}")

    print(f"{'Diffusion (ours)':22s} {mp:10.2f}  {ms:8.4f}")
    print(f"{'Improvement over input':22s} {mp - ip:+10.2f}  {ms - iss_:+8.4f}")

    if psnr_p2p:
        print(f" {'Improvement over p2p':22s} {mp - pp:+10.2f}  {ms - ps:+8.4f}")

    print(f"\n  Val slices: {len(psnr_diff)}")
    if GENERATE_SAMPLES:
        print(f" Saved {sample_idx} comparison images to {RESULTS_DIR}")

    # Save metrics
    import json
    metrics = {
        "psnr_diffusion": round(mp, 2),
        "ssim_diffusion": round(ms, 4),
        "psnr_lowdose": round(ip, 2),
        "ssim_lowdose": round(iss_, 4),
        "psnr_diffusion_std": round(float(np.std(psnr_diff)), 2),
        "ssim_diffusion_std": round(float(np.std(ssim_diff)), 4),
    }
    if psnr_p2p:
        metrics["psnr_pix2pix"] = round(pp, 2)
        metrics["ssim_pix2pix"] = round(ps, 4)

    with open(f"{RESULTS_DIR}/val_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    vol.commit()
    print(f"\nDownload: modal volume get ldct-data /results/diffusion_v2/validation_samples ./validation_samples")


@app.local_entrypoint()
def main():
    eval_val.remote()