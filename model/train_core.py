"""
train_core.py

The canonical training logic for the rectified flow model.
Plain function (no Modal decorator) so it can be called from either:
    - train.py            (standalone training)
    - full_pipeline.py    (as the final step in the chained pipeline)

Includes EMA, dropout regularization, and early stopping.
"""

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.latent_dataset import LDCTLatentDataset
from model.rectified_flow import RectifiedFlow
from model.unet import ConditionalUNet


def run_training(
    latent_dir="/data/latents",
    results_dir="/data/results/diffusion_v2",
    vol=None,                  # Modal volume to commit (optional)
    batch_size=16,
    num_epochs=80,
    lr=1e-4,
    dropout=0.15,
    ema_decay=0.999,
    patience=15,               # early stopping patience
    sample_steps=20,
    device=None,
):
    """
    Train the conditional rectified flow model.

    Returns the best validation loss achieved.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/samples", exist_ok=True)
    os.makedirs(f"{results_dir}/checkpoints", exist_ok=True)

    # --- Data ---
    train_ds = LDCTLatentDataset(latent_dir, split="train")
    val_ds = LDCTLatentDataset(latent_dir, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # --- Model ---
    backbone = ConditionalUNet(
        in_channels=4, cond_channels=4, base_channels=128,
        channel_mults=(1, 2, 4, 4), num_res_blocks=3,
        dropout=dropout, attn_resolutions=(8, 16),
        cross_attn_resolutions=(8, 16), n_heads=4,
    ).to(device)

    flow = RectifiedFlow(backbone).to(device)
    optimizer = torch.optim.AdamW(backbone.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # --- EMA (exponential moving average of weights) ---
    ema_state = {k: v.clone() for k, v in backbone.state_dict().items()}

    def update_ema():
        with torch.no_grad():
            for k, v in backbone.state_dict().items():
                if ema_state[k].dtype.is_floating_point:
                    ema_state[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)
                else:
                    ema_state[k].copy_(v)

    n_params = sum(p.numel() for p in backbone.parameters()) / 1e6
    print(f"Backbone: {n_params:.1f}M params")
    print(f"Training: {len(train_ds)} slices, {num_epochs} epochs, "
          f"batch_size={batch_size}, dropout={dropout}, ema={ema_decay}")

    # --- Training loop ---
    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        backbone.train()
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            z_cond = batch["z_low"].to(device)
            z_target = batch["z_full"].to(device)

            loss = flow.compute_loss(z_target=z_target, z_condition=z_cond)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
            optimizer.step()
            update_ema()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Validate
        backbone.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                z_cond = batch["z_low"].to(device)
                z_target = batch["z_full"].to(device)
                loss = flow.compute_loss(z_target=z_target, z_condition=z_cond)
                val_loss += loss.item()
                n_val += 1

        avg_val_loss = val_loss / n_val
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train: {avg_train_loss:.5f}  Val: {avg_val_loss:.5f} | "
              f"LR: {current_lr:.2e}")

        scheduler.step()
        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)

        # Save best (EMA weights)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "backbone_state": ema_state,
                "backbone_state_raw": backbone.state_dict(),
                "val_loss": avg_val_loss,
            }, f"{results_dir}/checkpoints/best.pt")
            print(f"  → New best (val_loss={avg_val_loss:.5f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

        # Periodic checkpoint + sample images
        if epoch % 20 == 0 or epoch == 1:
            torch.save({
                "epoch": epoch, "backbone_state": ema_state, "history": history,
            }, f"{results_dir}/checkpoints/epoch_{epoch:03d}.pt")

            _save_samples(backbone, flow, val_loader, ema_state, device,
                          sample_steps, results_dir, epoch)

            if vol is not None:
                vol.commit()

    # --- Final outputs ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train")
    ax.plot(history["epoch"], history["val_loss"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE on velocity)")
    ax.set_title("Rectified Flow Training")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/training_curves.png", dpi=120)
    plt.close()

    with open(f"{results_dir}/metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    if vol is not None:
        vol.commit()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.5f}")
    return best_val_loss


def _save_samples(backbone, flow, val_loader, ema_state, device,
                  sample_steps, results_dir, epoch):
    """Generate sample comparison images using EMA weights."""
    backbone.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        z_cond = sample_batch["z_low"][:4].to(device)
        z_target = sample_batch["z_full"][:4].to(device)

        # Swap in EMA weights for sampling
        raw_state = backbone.state_dict()
        backbone.load_state_dict(ema_state)
        z_pred = flow.sample(z_cond, num_steps=sample_steps, device=device)
        backbone.load_state_dict(raw_state)

        from model.vae import MedVAEWrapper
        vae = MedVAEWrapper(model_name="medvae_8_4_2d", device=device)
        pred_imgs = vae.decode(z_pred)
        target_imgs = vae.decode(z_target)
        cond_imgs = vae.decode(z_cond)
        del vae
        torch.cuda.empty_cache()

    n_s = min(4, z_cond.size(0))
    fig, axes = plt.subplots(3, n_s, figsize=(5 * n_s, 15))
    for i in range(n_s):
        axes[0, i].imshow(cond_imgs[i, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
        axes[0, i].set_title("Low-dose", fontsize=10)
        axes[0, i].axis("off")
        axes[1, i].imshow(pred_imgs[i, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
        axes[1, i].set_title("Diffusion output", fontsize=10)
        axes[1, i].axis("off")
        axes[2, i].imshow(target_imgs[i, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
        axes[2, i].set_title("Full-dose (GT)", fontsize=10)
        axes[2, i].axis("off")
    plt.suptitle(f"Epoch {epoch}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/samples/epoch_{epoch:03d}.png", dpi=120)
    plt.close()