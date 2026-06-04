"""
train.py

Train the conditional rectified flow model for low-dose → full-dose CT.

Uses pre-encoded MedVAE latents (from pre_encode.py) so training
operates entirely on (4, 64, 64) tensors — no VAE overhead per batch.

VAE is only loaded for decoding sample images every N epochs.

Run:     modal run train.py
Detach:  modal run --detach train.py
Results: modal volume get ldct-data /results/diffusion ./results_diffusion
"""

import modal
import os

app = modal.App("ldct-diffusion")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "medvae",
    )
    .add_local_python_source("data")
    .add_local_python_source("model")
)

vol = modal.Volume.from_name("ldct-data")


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="A100",
    timeout=14400,
)
def train():
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

    # --------------------Config-----------------------
    LATENT_DIR = "/data/latents"
    RESULTS_DIR = "/data/results/diffusion"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/samples", exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/checkpoints", exist_ok=True)

    BATCH_SIZE = 16    # Latents are tiny — can use bigger batches
    NUM_EPOCHS = 100
    LR = 1e-4
    SAMPLE_STEPS = 20
    SAVE_EVERY = 20
    SAMPLE_EVERY = 10

    device = torch.device("cuda")

    # Data (pre-encoded latents, done need to pass through vae again
    print("Loading pre-encoded latents")
    train_ds = LDCTLatentDataset(LATENT_DIR, split="train")
    val_ds = LDCTLatentDataset(LATENT_DIR, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    #------------- Model-----------------------
    print("Initializing backbone.")
    backbone = ConditionalUNet(
        in_channels=4,
        cond_channels=4,
        base_channels=128,
        channel_mults=(1, 2, 4, 4),
        num_res_blocks=2,
        dropout=0.1,
        attn_resolutions=(8,),
    ).to(device)

    flow = RectifiedFlow(backbone).to(device)

    optimizer = torch.optim.AdamW(backbone.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    n_params = sum(p.numel() for p in backbone.parameters()) / 1e6
    print(f"Backbone: {n_params:.1f}M params")
    print(f"Training: {len(train_ds)} slices, {NUM_EPOCHS} epochs, "
          f"batch_size={BATCH_SIZE}")

    #----------------Training loop ----------------------
    history = {
        "epoch": [], "train_loss": [], "val_loss": [], "lr": [],
    }
    best_val_loss = float("inf")

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---train ---
        backbone.train()
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            z_cond = batch["z_low"].to(device)     # (B, 4, 64, 64)
            z_target = batch["z_full"].to(device)   # (B, 4, 64, 64)

            loss = flow.compute_loss(z_target=z_target, z_condition=z_cond)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # ---validate ---
        backbone.eval()
        val_loss = 0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                z_cond = batch["z_low"].to(device)
                z_target = batch["z_full"].to(device)
                loss = flow.compute_loss(z_target=z_target, z_condition=z_cond)
                val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / n_val_batches
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"Train: {avg_train_loss:.5f}  Val: {avg_val_loss:.5f} | "
              f"LR: {current_lr:.2e}")

        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)

        # --- Save best model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "backbone_state": backbone.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": avg_val_loss,
            }, f"{RESULTS_DIR}/checkpoints/best.pt")
            print(f"  → New best (val_loss={avg_val_loss:.5f})")

        # --- Save checkpoint ---
        if epoch % SAVE_EVERY == 0 or epoch == NUM_EPOCHS:
            torch.save({
                "epoch": epoch,
                "backbone_state": backbone.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history,
            }, f"{RESULTS_DIR}/checkpoints/epoch_{epoch:03d}.pt")

        # --- Generate sample images (decode with VAE) ---
        if epoch % SAMPLE_EVERY == 0 or epoch == 1:
            backbone.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                z_cond = sample_batch["z_low"][:4].to(device)
                z_target = sample_batch["z_full"][:4].to(device)

                z_pred = flow.sample(z_cond, num_steps=SAMPLE_STEPS, device=device)

                # Load VAE just for decoding samples
                from model.vae import MedVAEWrapper
                vae = MedVAEWrapper(model_name="medvae_8_4_2d", device=device)

                pred_imgs = vae.decode(z_pred)
                target_imgs = vae.decode(z_target)
                cond_imgs = vae.decode(z_cond)

                del vae  # free VRAM

            n_samples = min(4, z_cond.size(0))
            fig, axes = plt.subplots(3, n_samples, figsize=(5 * n_samples, 15))

            for i in range(n_samples):
                axes[0, i].imshow(cond_imgs[i, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
                axes[0, i].set_title("Low-dose (decoded)", fontsize=10)
                axes[0, i].axis("off")

                axes[1, i].imshow(pred_imgs[i, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
                axes[1, i].set_title("Diffusion output", fontsize=10)
                axes[1, i].axis("off")

                axes[2, i].imshow(target_imgs[i, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
                axes[2, i].set_title("Full-dose (decoded)", fontsize=10)
                axes[2, i].axis("off")

            plt.suptitle(f"Epoch {epoch} | Val Loss: {avg_val_loss:.5f}", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{RESULTS_DIR}/samples/epoch_{epoch:03d}.png",
                        dpi=120, bbox_inches="tight")
            plt.close()

        # Commit volume periodically
        if epoch % SAVE_EVERY == 0:
            vol.commit()

    # --------Save final results-------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE on velocity)")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["lr"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("LR Schedule")

    plt.suptitle("Rectified Flow Training", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/training_curves.png", dpi=120)
    plt.close()

    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    vol.commit()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best val loss: {best_val_loss:.5f}")
    print(f"\nResults: {RESULTS_DIR}")
    print("Download:")
    print("  modal volume get ldct-data /results/diffusion ./results_diffusion")


@app.local_entrypoint()
def main():
    train.remote()