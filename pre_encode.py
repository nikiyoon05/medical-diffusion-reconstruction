"""
pre_encode.py

Pre-encode all CT images through MedVAE and save latents to the volume.
This eliminates the VAE bottleneck during training.

Input:  /data/processed/CXXX/{low,full}_dose/NNN.npy   (512×512 images)
Output: /data/latents/CXXX/{low,full}_dose/NNN.npy     (64×64×4 latents)

Run: modal run pre_encode.py
"""

import modal
import os

app = modal.App("ldct-pre-encode")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "medvae")
)

vol = modal.Volume.from_name("ldct-data")


@app.function(
    image=image,
    volumes={"/data": vol},
    gpu="A100",
    timeout=3600,
)
def pre_encode():
    import torch
    import numpy as np
    from medvae import MVAE

    device = torch.device("cuda")

    # Load MedVAE
    print("Loading MedVAE...")
    model = MVAE(model_name="medvae_8_4_2d", modality="ct").to(device)
    model.requires_grad_(False)
    model.eval()

    processed_dir = "/data/processed"
    latent_dir = "/data/latents"

    patients = sorted([
        d for d in os.listdir(processed_dir)
        if os.path.isdir(os.path.join(processed_dir, d))
    ])

    print(f"Found {len(patients)} patients")
    total = 0

    for pid in patients:
        for dose in ["low_dose", "full_dose"]:
            in_dir = os.path.join(processed_dir, pid, dose)
            out_dir = os.path.join(latent_dir, pid, dose)
            os.makedirs(out_dir, exist_ok=True)

            files = sorted([f for f in os.listdir(in_dir) if f.endswith(".npy")])

            for f in files:
                out_path = os.path.join(out_dir, f)
                if os.path.exists(out_path):
                    continue  # skip already encoded

                # Load image: (512, 512) float32 [0, 1]
                img = np.load(os.path.join(in_dir, f))

                # Convert to tensor: (1, 1, 512, 512) then repeat to 3ch
                tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
                tensor = tensor * 2 - 1  # [0,1] → [-1,1]
                tensor = tensor.repeat(1, 3, 1, 1)  # 1ch → 3ch for MedVAE

                with torch.no_grad():
                    latent = model.encode(tensor)  # (1, 4, 64, 64)

                # Save as numpy
                np.save(out_path, latent[0].cpu().numpy())  # (4, 64, 64)
                total += 1

            print(f"  {pid}/{dose}: {len(files)} slices encoded")

    vol.commit()

    print(f"\nDone! {total} latents saved to {latent_dir}")
    print(f"Each latent: (4, 64, 64) float32 = 64KB")

    # Verify
    sample = np.load(os.path.join(latent_dir, patients[0], "full_dose", "000.npy"))
    print(f"Sample shape: {sample.shape}, range: [{sample.min():.2f}, {sample.max():.2f}]")


@app.local_entrypoint()
def main():
    pre_encode.remote()