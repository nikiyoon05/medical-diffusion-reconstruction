"""
train.py

Standalone training for the rectified flow model.
All logic lives in model/train_core.py — this is just the Modal wrapper.

Run:     modal run --detach train.py
Results: modal volume get ldct-data /results/diffusion_v2 ./results_v2
"""

import modal

app = modal.App("ldct-diffusion")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "numpy", "matplotlib", "medvae")
    .add_local_python_source("data")
    .add_local_python_source("model")
)

vol = modal.Volume.from_name("ldct-data")


@app.function(image=image, volumes={"/data": vol}, gpu="A100", timeout=14400)
def train():
    from model.train_core import run_training
    run_training(
        latent_dir="/data/latents",
        results_dir="/data/results/diffusion_v2",
        vol=vol,
        num_epochs=80,
    )


@app.local_entrypoint()
def main():
    train.remote()