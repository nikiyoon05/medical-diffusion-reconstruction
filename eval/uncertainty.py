"""
uncertainty.py

Generate uncertainty heatmaps for CT reconstructions.

For each input slice:
    1. Run the SDE sampler 10 times --> 10 slightly different outputs
    2. Compute per pixel variance across the 10 runs
    3. Decode variance map through medvae --> 512×512 heatmap
    4. Overlay on the mean reconstruction

Output: 4-column images
    [Low-dose | Mean reconstruction | Uncertainty heatmap | Full-dose GT]

Run: modal run uncertainty.py
Get: modal volume get ldct-data /results/diffusion_v2/uncertainty ./uncertainty
"""

import modal

app = modal.App("ldct-uncertainty")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "matplotlib", "medvae")
    .add_local_python_source("data")
    .add_local_python_source("model")
)

vol = modal.Volume.from_name("ldct-data")

# Config
CHECKPOINT = "/data/results/diffusion_v2/checkpoints/best.pt"
N_RUNS = 10  # Number of stochastic samples
SAMPLE_STEPS = 50   # Steps per SDE run
SIGMA = 0.15   # Noise strength 
N_SLICES = 8  # Number of slices to visualize


@app.function(image=image, volumes={"/data": vol}, gpu="A100", timeout=3600)
def generate_uncertainty():
    import torch
    import numpy as np
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from torch.utils.data import DataLoader
    from data.latent_dataset import LDCTLatentDataset
    from model.rectified_flow import RectifiedFlow
    from model.unet import ConditionalUNet
    from model.vae import MedVAEWrapper

    device = torch.device("cuda")

    RESULTS_DIR = "/data/results/diffusion_v2/uncertainty"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load model
    print("Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT, map_location=device)
    print(f"  Epoch {ckpt['epoch']}")

    backbone = ConditionalUNet(
        in_channels=4, cond_channels=4, base_channels=128,
        channel_mults=(1, 2, 4, 4), num_res_blocks=2,
        dropout=0.0, attn_resolutions=(8,),
    ).to(device)
    backbone.load_state_dict(ckpt["backbone_state"])
    backbone.eval()

    flow = RectifiedFlow(backbone).to(device)
    vae = MedVAEWrapper(model_name="medvae_8_4_2d", device=device)

    # Val set
    val_ds = LDCTLatentDataset("/data/latents", split="val", augment=False)
    # Pick evenly spaced slices for variety
    indices = np.linspace(0, len(val_ds) - 1, N_SLICES, dtype=int)

    print(f"\nGenerating uncertainty maps ({N_RUNS} runs × {SAMPLE_STEPS} steps)...")
    print(f"Sigma: {SIGMA}")

    for slice_num, idx in enumerate(indices):
        sample = val_ds[idx]
        z_cond = sample["z_low"].unsqueeze(0).to(device)    # (1, 4, 64, 64)
        z_target = sample["z_full"].unsqueeze(0).to(device)

        print(f"\n  Slice {slice_num + 1}/{N_SLICES} (idx={idx}):")
        print(f"    Running {N_RUNS} SDE samples...", end=" ", flush=True)

        # Compute uncertainty in latent space
        mean_latent, var_latent, all_samples = flow.compute_uncertainty(
            z_cond, n_runs=N_RUNS, num_steps=SAMPLE_STEPS,
            sigma=SIGMA, device=device,
        )
        print("done")

        # Decode to image space
        with torch.no_grad():
            mean_img = vae.decode(mean_latent) # (1, 1, 512, 512)
            cond_img = vae.decode(z_cond)     # low-dose
            target_img = vae.decode(z_target)  # ground truth

            # Decode each sample and compute image-space variance
            decoded_samples = []
            for s in all_samples:
                decoded_samples.append(vae.decode(s))

        # Stack decoded samples: (N_RUNS, 1, 1, 512, 512)
        decoded_stack = torch.stack(decoded_samples, dim=0)
        img_variance = decoded_stack.var(dim=0)  # (1, 1, 512, 512)

        # Convert to numpy
        mean_np = mean_img[0, 0].cpu().numpy()
        cond_np = cond_img[0, 0].cpu().numpy()
        target_np = target_img[0, 0].cpu().numpy()
        var_np = img_variance[0, 0].cpu().numpy()

        # Normalize variance for visualization
        var_normalized = var_np / (var_np.max() + 1e-8)

        # Fig 1: Full comparison 
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        # low dose input
        axes[0].imshow(cond_np, cmap="gray", vmin=-1, vmax=1)
        axes[0].set_title("Low-dose input", fontsize=13)
        axes[0].axis("off")

        # mean reconstruction
        axes[1].imshow(mean_np, cmap="gray", vmin=-1, vmax=1)
        axes[1].set_title(f"Mean reconstruction\n({N_RUNS} SDE runs)", fontsize=13)
        axes[1].axis("off")

        # Uncertainty heatmap overlaid on reconstruction
        axes[2].imshow(mean_np, cmap="gray", vmin=-1, vmax=1)
        heatmap = axes[2].imshow(var_normalized, cmap="hot", alpha=0.6,
                                  vmin=0, vmax=0.5)
        axes[2].set_title("Uncertainty heatmap\n(red = low confidence)", fontsize=13)
        axes[2].axis("off")
        plt.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04,
                      label="Relative variance")

        # Ground truth
        axes[3].imshow(target_np, cmap="gray", vmin=-1, vmax=1)
        axes[3].set_title("Full-dose (GT)", fontsize=13)
        axes[3].axis("off")

        plt.suptitle(f"Uncertainty Estimation — σ={SIGMA}, {N_RUNS} runs",
                      fontsize=15, y=1.02)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/uncertainty_{slice_num:03d}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        # === Figure 2: Uncertainty vs actual error ===
        actual_error = np.abs(mean_np - target_np)
        error_normalized = actual_error / (actual_error.max() + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(var_normalized, cmap="hot", vmin=0, vmax=0.5)
        axes[0].set_title("Predicted uncertainty\n(variance heatmap)", fontsize=13)
        axes[0].axis("off")

        axes[1].imshow(error_normalized, cmap="hot", vmin=0, vmax=0.5)
        axes[1].set_title("Actual reconstruction error\n(|pred - GT|)", fontsize=13)
        axes[1].axis("off")

        # Correlation: scatter plot of variance vs actual error
        # Downsample for visualization
        step = 8
        var_flat = var_normalized[::step, ::step].flatten()
        err_flat = error_normalized[::step, ::step].flatten()
        corr = np.corrcoef(var_flat, err_flat)[0, 1]

        axes[2].scatter(var_flat, err_flat, alpha=0.1, s=1, color="tab:red")
        axes[2].set_xlabel("Predicted uncertainty", fontsize=12)
        axes[2].set_ylabel("Actual error", fontsize=12)
        axes[2].set_title(f"Correlation: r = {corr:.3f}", fontsize=13)
        axes[2].set_xlim(0, 0.5)
        axes[2].set_ylim(0, 0.5)

        plt.suptitle("Does uncertainty predict actual error?", fontsize=15, y=1.02)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/correlation_{slice_num:03d}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        print(f" Variance range: [{var_np.min():.6f}, {var_np.max():.6f}]")
        print(f"Uncertainty-error correlation: r = {corr:.3f}")

    vol.commit()
    print(f"\nSaved to {RESULTS_DIR}")
    print("Download: modal volume get ldct-data /results/diffusion_v2/uncertainty ./uncertainty")


@app.local_entrypoint()
def main():
    generate_uncertainty.remote()