"""
eval_diffusion.py

Evaluate the trained diffusion model on the test set.
Loads the best checkpoint (EMA weights), samples, decodes, and computes
PSNR / SSIM against ground truth, plus comparison images.

Run: modal run eval_diffusion.py
Results: modal volume get ldct-data /results/diffusion_v2/eval ./eval_diffusion
"""

import modal

app = modal.App("ldct-eval-diffusion")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "numpy", "matplotlib", "medvae")
    .add_local_python_source("data")
    .add_local_python_source("model")
)

vol = modal.Volume.from_name("ldct-data")

RESULTS_DIR = "/data/results/diffusion_v2"
SAMPLE_STEPS = 50  # more steps at eval time for best quality


@app.function(image=image, volumes={"/data": vol}, gpu="A100", timeout=3600)
def evaluate():
    import os
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import json
    from torch.utils.data import DataLoader

    from data.latent_dataset import LDCTLatentDataset
    from model.rectified_flow import RectifiedFlow
    from model.unet import ConditionalUNet
    from model.vae import MedVAEWrapper

    device = torch.device("cuda")
    eval_dir = f"{RESULTS_DIR}/eval"
    os.makedirs(f"{eval_dir}/comparisons", exist_ok=True)

    # --- Metrics ---
    def psnr(pred, target):
        mse = torch.mean((pred - target) ** 2).item()
        return float("inf") if mse == 0 else 10 * np.log10(4.0 / mse)

    def ssim(pred, target, window=11):
        C1, C2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2
        sigma = 1.5
        coords = torch.arange(window, dtype=torch.float32) - window // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g = g / g.sum()
        k = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0).to(pred.device)
        mu1 = torch.nn.functional.conv2d(pred, k, padding=window // 2)
        mu2 = torch.nn.functional.conv2d(target, k, padding=window // 2)
        m1, m2, m12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        s1 = torch.nn.functional.conv2d(pred ** 2, k, padding=window // 2) - m1
        s2 = torch.nn.functional.conv2d(target ** 2, k, padding=window // 2) - m2
        s12 = torch.nn.functional.conv2d(pred * target, k, padding=window // 2) - m12
        smap = ((2 * m12 + C1) * (2 * s12 + C2)) / ((m1 + m2 + C1) * (s1 + s2 + C2))
        return smap.mean().item()

    # --- Load model with EMA weights ---
    print("Loading best checkpoint...")
    ckpt = torch.load(f"{RESULTS_DIR}/checkpoints/best.pt", map_location=device)
    print(f"  Best epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.5f}")

    backbone = ConditionalUNet(
        in_channels=4, cond_channels=4, base_channels=128,
        channel_mults=(1, 2, 4, 4), num_res_blocks=2,
        dropout=0.15, attn_resolutions=(8,),
    ).to(device)
    backbone.load_state_dict(ckpt["backbone_state"])  # EMA weights
    backbone.eval()
    flow = RectifiedFlow(backbone).to(device)

    vae = MedVAEWrapper(model_name="medvae_8_4_2d", device=device)

    # --- Test set ---
    test_ds = LDCTLatentDataset("/data/latents", split="test")
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    all_psnr_model, all_ssim_model = [], []
    all_psnr_input, all_ssim_input = [], []

    print(f"Evaluating on {len(test_ds)} test slices ({SAMPLE_STEPS} sampling steps)...")
    with torch.no_grad():
        for batch in test_loader:
            z_cond = batch["z_low"].to(device)
            z_target = batch["z_full"].to(device)

            z_pred = flow.sample(z_cond, num_steps=SAMPLE_STEPS, device=device)

            pred_imgs = vae.decode(z_pred)
            target_imgs = vae.decode(z_target)
            input_imgs = vae.decode(z_cond)

            for i in range(pred_imgs.size(0)):
                all_psnr_model.append(psnr(pred_imgs[i:i+1], target_imgs[i:i+1]))
                all_ssim_model.append(ssim(pred_imgs[i:i+1], target_imgs[i:i+1]))
                all_psnr_input.append(psnr(input_imgs[i:i+1], target_imgs[i:i+1]))
                all_ssim_input.append(ssim(input_imgs[i:i+1], target_imgs[i:i+1]))

    mp, ms = np.mean(all_psnr_model), np.mean(all_ssim_model)
    ip, is_ = np.mean(all_psnr_input), np.mean(all_ssim_input)

    print("\n" + "=" * 60)
    print("DIFFUSION TEST RESULTS")
    print("=" * 60)
    print(f"  {'':22s} {'PSNR (dB)':>10s}  {'SSIM':>8s}")
    print(f"  {'Low-dose (no model)':22s} {ip:10.2f}  {is_:8.4f}")
    print(f"  {'Diffusion (ours)':22s} {mp:10.2f}  {ms:8.4f}")
    print(f"  {'Improvement':22s} {mp - ip:+10.2f}  {ms - is_:+8.4f}")
    print(f"\n  Test slices: {len(all_psnr_model)}")

    # --- Comparison images (first 12 test slices) ---
    print("\nGenerating comparison images...")
    single = DataLoader(test_ds, batch_size=1, shuffle=False)
    with torch.no_grad():
        for idx, batch in enumerate(single):
            if idx >= 12:
                break
            z_cond = batch["z_low"].to(device)
            z_target = batch["z_full"].to(device)
            z_pred = flow.sample(z_cond, num_steps=SAMPLE_STEPS, device=device)

            pred = vae.decode(z_pred)[0, 0].cpu().numpy()
            tgt = vae.decode(z_target)[0, 0].cpu().numpy()
            inp = vae.decode(z_cond)[0, 0].cpu().numpy()
            resid = np.abs(pred - tgt)

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(inp, cmap="gray", vmin=-1, vmax=1)
            axes[0].set_title("Low-dose"); axes[0].axis("off")
            axes[1].imshow(pred, cmap="gray", vmin=-1, vmax=1)
            axes[1].set_title(f"Diffusion (PSNR {psnr(torch.tensor(pred)[None,None], torch.tensor(tgt)[None,None]):.1f})")
            axes[1].axis("off")
            axes[2].imshow(tgt, cmap="gray", vmin=-1, vmax=1)
            axes[2].set_title("Full-dose (GT)"); axes[2].axis("off")
            axes[3].imshow(resid, cmap="hot", vmin=0, vmax=0.5)
            axes[3].set_title("Residual"); axes[3].axis("off")
            plt.tight_layout()
            plt.savefig(f"{eval_dir}/comparisons/slice_{idx:03d}.png", dpi=120,
                        bbox_inches="tight")
            plt.close()

    # --- Save metrics ---
    summary = {
        "model": "diffusion_rectified_flow",
        "best_epoch": ckpt["epoch"],
        "sample_steps": SAMPLE_STEPS,
        "test_slices": len(all_psnr_model),
        "psnr_model": round(float(mp), 2),
        "ssim_model": round(float(ms), 4),
        "psnr_input": round(float(ip), 2),
        "ssim_input": round(float(is_), 4),
    }
    with open(f"{eval_dir}/test_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    vol.commit()
    print(f"\nResults: {eval_dir}")
    print("Download: modal volume get ldct-data /results/diffusion_v2/eval ./eval_diffusion")


@app.local_entrypoint()
def main():
    evaluate.remote()