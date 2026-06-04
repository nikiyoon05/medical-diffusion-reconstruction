"""
smoke_test.py

Verifies the entire training pipeline works end to end:
    dataset → MedVAE encode → rectified flow → backbone → loss → backward

Run: modal run smoke_test.py
"""

import modal
import os

app = modal.App("ldct-smoke-test")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "medvae",
    )
    .add_local_python_source("data")
    .add_local_python_source("model")
)

vol = modal.Volume.from_name("ldct-data")


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="A10G",
    timeout=600,
)
def smoke_test():
    import torch
    from torch.utils.data import DataLoader
    from data.dataset import LDCTDataset
    from model.vae import MedVAEWrapper
    from model.rectified_flow import RectifiedFlow
    from model.unet import ConditionalUNet

    device = torch.device("cuda")

    print("=" * 60)
    print("SMOKE TEST: Full pipeline check")
    print("=" * 60)

    # --- 1. Dataset ---
    print("\n[1/6] Loading dataset...")
    ds = LDCTDataset("/data/processed", split="train", img_size=512)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    batch = next(iter(loader))

    low = batch["low_dose"].to(device)
    full = batch["full_dose"].to(device)
    print(f"  low_dose:  {low.shape}, range [{low.min():.2f}, {low.max():.2f}]")
    print(f"  full_dose: {full.shape}, range [{full.min():.2f}, {full.max():.2f}]")

    # --- 2. MedVAE encode ---
    print("\n[2/6] MedVAE encode...")
    vae = MedVAEWrapper(model_name="medvae_8_4_2d", device=device)

    with torch.no_grad():
        z_cond = vae.encode(low)
        z_target = vae.encode(full)

    print(f"  z_cond:   {z_cond.shape}, range [{z_cond.min():.2f}, {z_cond.max():.2f}]")
    print(f"  z_target: {z_target.shape}, range [{z_target.min():.2f}, {z_target.max():.2f}]")

    # --- 3. MedVAE decode (roundtrip test) ---
    print("\n[3/6] MedVAE decode roundtrip...")
    with torch.no_grad():
        recon = vae.decode(z_target)
    print(f"  decoded: {recon.shape}, range [{recon.min():.2f}, {recon.max():.2f}]")

    mse = torch.mean((recon - full) ** 2).item()
    print(f"  roundtrip MSE: {mse:.6f} (lower = better VAE reconstruction)")

    # --- 4. Backbone forward ---
    print("\n[4/6] Backbone forward pass...")
    backbone = ConditionalUNet(
        in_channels=vae.latent_channels,
        cond_channels=vae.latent_channels,
        base_channels=128,
    ).to(device)

    n_params = sum(p.numel() for p in backbone.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f}M")

    t = torch.rand(2, device=device)
    noise = torch.randn_like(z_target)
    x_t = 0.5 * noise + 0.5 * z_target

    v_pred = backbone(x_t, t, z_cond)
    print(f"  Input:  x_t={x_t.shape}, t={t.shape}, cond={z_cond.shape}")
    print(f"  Output: v_pred={v_pred.shape}")
    assert v_pred.shape == z_target.shape, "Shape mismatch!"

    # --- 5. Rectified flow loss ---
    print("\n[5/6] Rectified flow loss...")
    flow = RectifiedFlow(backbone).to(device)
    loss = flow.compute_loss(z_target=z_target, z_condition=z_cond)
    print(f"  Loss: {loss.item():.6f}")

    # --- 6. Backward pass ---
    print("\n[6/6] Backward pass...")
    loss.backward()

    has_grad = backbone.input_conv.weight.grad is not None
    grad_norm = torch.nn.utils.clip_grad_norm_(backbone.parameters(), float("inf"))
    print(f"  Gradients flowing: {has_grad}")
    print(f"  Gradient norm: {grad_norm:.4f}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
    print(f"  Dataset:        {len(ds)} training slices")
    print(f"  Image shape:    {low.shape}")
    print(f"  Latent shape:   {z_cond.shape}")
    print(f"  Backbone:       {n_params:.1f}M params")
    print(f"  Loss:           {loss.item():.6f}")
    print(f"  VAE roundtrip:  MSE={mse:.6f}")
    print(f"\nReady to train! Run:")
    print(f"  modal run --detach train.py")


@app.local_entrypoint()
def main():
    smoke_test.remote()