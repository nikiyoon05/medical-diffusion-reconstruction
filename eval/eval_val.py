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
    .add_local_python_source("baselines")
)
 
vol = modal.Volume.from_name("ldct-data")
 
# ----CONFIG-----
CHECKPOINT = "/data/results/diffusion_v3/checkpoints/best.pt"
PIX2PIX_CHECKPOINT = "/data/results/pix2pix_v2/checkpoints/epoch_035.pt"
SAMPLE_STEPS = 100
GENERATE_SAMPLES = True
COMPARE_PIX2PIX = True
MAX_SAMPLES = 16
 
 
@app.function(image=image, volumes={"/data": vol}, gpu="H100", timeout=10000)
def eval_val():
    import torch
    import torch.nn as nn
    import numpy as np
    import os
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    from torch.utils.data import DataLoader
    from data.latent_dataset import LDCTLatentDataset
    from data.dataset import LDCTDataset
    from model.rectified_flow import RectifiedFlow
    from model.unet import ConditionalUNet
    from model.vae import MedVAEWrapper
 
    device = torch.device("cuda")
 
    RESULTS_DIR = "/data/results/diffusion_v2/validation_samples"
    os.makedirs(RESULTS_DIR, exist_ok=True)
 
    # ------------------------------------------------------------------
    # Load diffusion model
    print("Loading diffusion checkpoint...")
    ckpt = torch.load(CHECKPOINT, map_location=device)
    print(f"  Epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.5f}")
 
    backbone = ConditionalUNet(
        in_channels=4, cond_channels=4, base_channels=128,
        channel_mults=(1, 2, 4, 4), num_res_blocks=3,
        dropout=0.0, attn_resolutions=(8, 16),
        cross_attn_resolutions=(8, 16), n_heads=4,
    ).to(device)
    backbone.load_state_dict(ckpt["backbone_state"])
    backbone.eval()
 
    flow = RectifiedFlow(backbone).to(device)
    vae = MedVAEWrapper(model_name="medvae_8_4_2d", device=device)
 
    # ------------------------------------------------------------------
    # Load pix2pix
    # ------------------------------------------------------------------
    pix2pix_gen = None
    if COMPARE_PIX2PIX and os.path.exists(PIX2PIX_CHECKPOINT):
        print("Loading pix2pix...")
        from baselines.pix2pix import UNetGenerator
        pix2pix_gen = UNetGenerator().to(device)
        p2p_ckpt = torch.load(PIX2PIX_CHECKPOINT, map_location=device)
        pix2pix_gen.load_state_dict(p2p_ckpt["gen_state"])
        pix2pix_gen.eval()
        print("  Pix2pix loaded")
    elif COMPARE_PIX2PIX:
        print(f"  WARNING: pix2pix checkpoint not found at {PIX2PIX_CHECKPOINT}")
 
    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    val_latent = LDCTLatentDataset("/data/latents", split="val", augment=False)
    val_loader = DataLoader(val_latent, batch_size=4, shuffle=False)
 
    val_images = None
    if pix2pix_gen is not None:
        val_images = LDCTDataset("/data/processed", split="val", img_size=256)
 
    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def compute_psnr(pred, target):
        mse = torch.mean((pred - target) ** 2).item()
        return float("inf") if mse == 0 else 10 * np.log10(4.0 / mse)
 
    def compute_mse(pred, target):
        return torch.mean((pred - target) ** 2).item()
 
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
 
    # ------------------------------------------------------------------
    # Identify which patient each slice belongs to
    # ------------------------------------------------------------------
    all_patients = sorted([
        d for d in os.listdir("/data/latents")
        if os.path.isdir(os.path.join("/data/latents", d))
    ])
    n = len(all_patients)
    val_patients = all_patients[int(0.70 * n):int(0.85 * n)]
    print(f"Val patients: {val_patients}")
 
    # Build a list mapping slice index → patient name
    slice_to_patient = []
    for patient in val_patients:
        ld_dir = os.path.join("/data/latents", patient, "low_dose")
        if os.path.isdir(ld_dir):
            n_slices = len([f for f in os.listdir(ld_dir) if f.endswith(".npy")])
            slice_to_patient.extend([patient] * n_slices)
 
    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    print(f"\nEval: {len(val_latent)} slices, {SAMPLE_STEPS} midpoint steps")
 
    results_diff = []   # per-slice: {patient, psnr, ssim, mse}
    results_input = []
    results_p2p = []
 
    n_done = 0
    sample_idx = 0
    sample_interval = max(1, len(val_latent) // MAX_SAMPLES)
 
    with torch.no_grad():
        for batch in val_loader:
            z_cond = batch["z_low"].to(device)
            z_target = batch["z_full"].to(device)
            B = z_cond.size(0)
 
            z_pred = flow.sample_midpoint(z_cond, num_steps=SAMPLE_STEPS, device=device)
 
            pred_imgs = vae.decode(z_pred)
            target_imgs = vae.decode(z_target)
            input_imgs = vae.decode(z_cond)
 
            # Pix2pix
            p2p_imgs = None
            if pix2pix_gen is not None and val_images is not None:
                p2p_list = []
                for j in range(B):
                    img_idx = n_done + j
                    if img_idx < len(val_images):
                        img_batch = val_images[img_idx]
                        low_256 = img_batch["low_dose"].unsqueeze(0).to(device)
                        p2p_out = pix2pix_gen(low_256)
                        p2p_out = nn.functional.interpolate(
                            p2p_out, size=512, mode="bilinear", align_corners=False)
                        p2p_list.append(p2p_out)
                if p2p_list:
                    p2p_imgs = torch.cat(p2p_list, dim=0)
 
            for i in range(B):
                global_idx = n_done + i
                patient = slice_to_patient[global_idx] if global_idx < len(slice_to_patient) else "unknown"
 
                d_psnr = compute_psnr(pred_imgs[i:i+1], target_imgs[i:i+1])
                d_ssim = compute_ssim(pred_imgs[i:i+1], target_imgs[i:i+1])
                d_mse = compute_mse(pred_imgs[i:i+1], target_imgs[i:i+1])
                results_diff.append({"patient": patient, "psnr": d_psnr, "ssim": d_ssim, "mse": d_mse})
 
                i_psnr = compute_psnr(input_imgs[i:i+1], target_imgs[i:i+1])
                i_ssim = compute_ssim(input_imgs[i:i+1], target_imgs[i:i+1])
                i_mse = compute_mse(input_imgs[i:i+1], target_imgs[i:i+1])
                results_input.append({"patient": patient, "psnr": i_psnr, "ssim": i_ssim, "mse": i_mse})
 
                if p2p_imgs is not None and i < p2p_imgs.size(0):
                    p_psnr = compute_psnr(p2p_imgs[i:i+1], target_imgs[i:i+1])
                    p_ssim = compute_ssim(p2p_imgs[i:i+1], target_imgs[i:i+1])
                    p_mse = compute_mse(p2p_imgs[i:i+1], target_imgs[i:i+1])
                    results_p2p.append({"patient": patient, "psnr": p_psnr, "ssim": p_ssim, "mse": p_mse})
 
                # --- Comparison image ---
                if GENERATE_SAMPLES and sample_idx < MAX_SAMPLES and global_idx % sample_interval == 0:
                    inp = input_imgs[i, 0].cpu().numpy()
                    diff = pred_imgs[i, 0].cpu().numpy()
                    gt = target_imgs[i, 0].cpu().numpy()
                    resid_diff = np.abs(diff - gt)
 
                    if p2p_imgs is not None and i < p2p_imgs.size(0):
                        p2p = p2p_imgs[i, 0].cpu().numpy()
                        resid_p2p = np.abs(p2p - gt)
 
                        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
 
                        axes[0, 0].imshow(inp, cmap="gray", vmin=-1, vmax=1)
                        axes[0, 0].set_title("Low-dose input", fontsize=12)
                        axes[0, 0].axis("off")
 
                        axes[0, 1].imshow(p2p, cmap="gray", vmin=-1, vmax=1)
                        axes[0, 1].set_title(
                            f"Pix2Pix\nPSNR={p_psnr:.1f}  SSIM={p_ssim:.3f}  MSE={p_mse:.4f}",
                            fontsize=11)
                        axes[0, 1].axis("off")
 
                        axes[0, 2].imshow(diff, cmap="gray", vmin=-1, vmax=1)
                        axes[0, 2].set_title(
                            f"Diffusion (ours)\nPSNR={d_psnr:.1f}  SSIM={d_ssim:.3f}  MSE={d_mse:.4f}",
                            fontsize=11)
                        axes[0, 2].axis("off")
 
                        axes[1, 0].imshow(gt, cmap="gray", vmin=-1, vmax=1)
                        axes[1, 0].set_title("Full-dose (GT)", fontsize=12)
                        axes[1, 0].axis("off")
 
                        axes[1, 1].imshow(resid_p2p, cmap="hot", vmin=0, vmax=0.5)
                        axes[1, 1].set_title("Pix2Pix residual error", fontsize=12)
                        axes[1, 1].axis("off")
 
                        axes[1, 2].imshow(resid_diff, cmap="hot", vmin=0, vmax=0.5)
                        axes[1, 2].set_title("Diffusion residual error", fontsize=12)
                        axes[1, 2].axis("off")
                    else:
                        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                        axes[0, 0].imshow(inp, cmap="gray", vmin=-1, vmax=1)
                        axes[0, 0].set_title("Low-dose input"); axes[0, 0].axis("off")
                        axes[0, 1].imshow(diff, cmap="gray", vmin=-1, vmax=1)
                        axes[0, 1].set_title(
                            f"Diffusion\nPSNR={d_psnr:.1f}  SSIM={d_ssim:.3f}  MSE={d_mse:.4f}")
                        axes[0, 1].axis("off")
                        axes[1, 0].imshow(gt, cmap="gray", vmin=-1, vmax=1)
                        axes[1, 0].set_title("Full-dose (GT)"); axes[1, 0].axis("off")
                        axes[1, 1].imshow(resid_diff, cmap="hot", vmin=0, vmax=0.5)
                        axes[1, 1].set_title("Residual error"); axes[1, 1].axis("off")
 
                    plt.suptitle(f"Patient {patient} — Slice {global_idx}", fontsize=14)
                    plt.tight_layout()
                    plt.savefig(f"{RESULTS_DIR}/val_{sample_idx:03d}.png",
                                dpi=150, bbox_inches="tight")
                    plt.close()
                    sample_idx += 1
 
            n_done += B
            if n_done % 100 == 0:
                print(f"  {n_done}/{len(val_latent)} — "
                      f"PSNR: {np.mean([r['psnr'] for r in results_diff]):.2f}  "
                      f"SSIM: {np.mean([r['ssim'] for r in results_diff]):.4f}")
 
    # ------------------------------------------------------------------
    # Summary metrics
    # ------------------------------------------------------------------
    mp = np.mean([r["psnr"] for r in results_diff])
    ms = np.mean([r["ssim"] for r in results_diff])
    mm = np.mean([r["mse"] for r in results_diff])
    ip = np.mean([r["psnr"] for r in results_input])
    iss_ = np.mean([r["ssim"] for r in results_input])
    im = np.mean([r["mse"] for r in results_input])
 
    print("\n" + "=" * 70)
    print(f"VALIDATION RESULTS (midpoint, {SAMPLE_STEPS} steps)")
    print("=" * 70)
    print(f"  {'':22s} {'PSNR (dB)':>10s}  {'SSIM':>8s}  {'MSE':>10s}")
    print(f"  {'Low-dose (no model)':22s} {ip:10.2f}  {iss_:8.4f}  {im:10.6f}")
 
    if results_p2p:
        pp = np.mean([r["psnr"] for r in results_p2p])
        ps = np.mean([r["ssim"] for r in results_p2p])
        pm = np.mean([r["mse"] for r in results_p2p])
        print(f"  {'Pix2Pix':22s} {pp:10.2f}  {ps:8.4f}  {pm:10.6f}")
 
    print(f"  {'Diffusion (ours)':22s} {mp:10.2f}  {ms:8.4f}  {mm:10.6f}")
    print(f"  {'Improvement over input':22s} {mp-ip:+10.2f}  {ms-iss_:+8.4f}  {im-mm:+10.6f}")
 
    if results_p2p:
        print(f"  {'Improvement over p2p':22s} {mp-pp:+10.2f}  {ms-ps:+8.4f}  {pm-mm:+10.6f}")
 
    # ------------------------------------------------------------------
    # Per-patient breakdown
    # ------------------------------------------------------------------
    print("\n  Per-patient breakdown (Diffusion):")
    per_patient = {}
    for r in results_diff:
        p = r["patient"]
        if p not in per_patient:
            per_patient[p] = {"psnr": [], "ssim": [], "mse": []}
        per_patient[p]["psnr"].append(r["psnr"])
        per_patient[p]["ssim"].append(r["ssim"])
        per_patient[p]["mse"].append(r["mse"])
 
    per_patient_summary = {}
    for p in sorted(per_patient.keys()):
        pp_psnr = np.mean(per_patient[p]["psnr"])
        pp_ssim = np.mean(per_patient[p]["ssim"])
        pp_mse = np.mean(per_patient[p]["mse"])
        n_sl = len(per_patient[p]["psnr"])
        print(f"    {p}: PSNR={pp_psnr:.2f}  SSIM={pp_ssim:.4f}  MSE={pp_mse:.6f}  ({n_sl} slices)")
        per_patient_summary[p] = {
            "psnr": round(pp_psnr, 2), "ssim": round(pp_ssim, 4),
            "mse": round(pp_mse, 6), "n_slices": n_sl
        }
 
    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    metrics = {
        "sample_steps": SAMPLE_STEPS,
        "val_slices": len(results_diff),
        "diffusion": {
            "psnr_mean": round(mp, 2), "psnr_std": round(float(np.std([r["psnr"] for r in results_diff])), 2),
            "ssim_mean": round(ms, 4), "ssim_std": round(float(np.std([r["ssim"] for r in results_diff])), 4),
            "mse_mean": round(mm, 6),
        },
        "lowdose": {
            "psnr_mean": round(ip, 2), "ssim_mean": round(iss_, 4), "mse_mean": round(im, 6),
        },
    }
    if results_p2p:
        metrics["pix2pix"] = {
            "psnr_mean": round(pp, 2), "psnr_std": round(float(np.std([r["psnr"] for r in results_p2p])), 2),
            "ssim_mean": round(ps, 4), "ssim_std": round(float(np.std([r["ssim"] for r in results_p2p])), 4),
            "mse_mean": round(pm, 6),
        }
 
    with open(f"{RESULTS_DIR}/val_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{RESULTS_DIR}/per_patient_metrics.json", "w") as f:
        json.dump(per_patient_summary, f, indent=2)
 
    # ------------------------------------------------------------------
    # Distribution histograms
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 
    axes[0].hist([r["psnr"] for r in results_diff], bins=40, alpha=0.7, label="Diffusion", color="tab:blue")
    if results_p2p:
        axes[0].hist([r["psnr"] for r in results_p2p], bins=40, alpha=0.7, label="Pix2Pix", color="tab:orange")
    axes[0].axvline(mp, color="tab:blue", linestyle="--", label=f"Diff mean={mp:.1f}")
    axes[0].set_xlabel("PSNR (dB)"); axes[0].set_title("PSNR Distribution")
    axes[0].legend()
 
    axes[1].hist([r["ssim"] for r in results_diff], bins=40, alpha=0.7, label="Diffusion", color="tab:blue")
    if results_p2p:
        axes[1].hist([r["ssim"] for r in results_p2p], bins=40, alpha=0.7, label="Pix2Pix", color="tab:orange")
    axes[1].set_xlabel("SSIM"); axes[1].set_title("SSIM Distribution")
    axes[1].legend()
 
    axes[2].hist([r["mse"] for r in results_diff], bins=40, alpha=0.7, label="Diffusion", color="tab:blue")
    if results_p2p:
        axes[2].hist([r["mse"] for r in results_p2p], bins=40, alpha=0.7, label="Pix2Pix", color="tab:orange")
    axes[2].set_xlabel("MSE"); axes[2].set_title("MSE Distribution")
    axes[2].legend()
 
    plt.suptitle("Metric Distributions (Validation Set)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/metric_distributions.png", dpi=150)
    plt.close()
 
    # ------------------------------------------------------------------
    # Per-patient bar chart
    # ------------------------------------------------------------------
    patients_sorted = sorted(per_patient.keys())
    x = np.arange(len(patients_sorted))
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
 
    for ax, metric, label in zip(axes, ["psnr", "ssim", "mse"], ["PSNR (dB)", "SSIM", "MSE"]):
        vals = [np.mean(per_patient[p][metric]) for p in patients_sorted]
        ax.bar(x, vals, color="tab:blue", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(patients_sorted, rotation=45, fontsize=9)
        ax.set_ylabel(label); ax.set_title(f"{label} by Patient")
        ax.axhline(np.mean(vals), color="red", linestyle="--", alpha=0.6, label="Mean")
        ax.legend()
 
    plt.suptitle("Per-Patient Performance (Diffusion)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/per_patient_chart.png", dpi=150)
    plt.close()
 
    vol.commit()
    print(f"\nSaved {sample_idx} comparison images + charts + metrics")
    print(f"Download: modal volume get ldct-data /results/diffusion_v2/validation_samples ./validation_samples")
 
 
@app.local_entrypoint()
def main():
    eval_val.remote()