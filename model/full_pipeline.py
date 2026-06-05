"""
full_pipeline.py

**THIS RETRAINS THE MODEL AND DOES ALL PREPROCESSING**
Runs all four steps in sequence inside a single Modal container so it
survives logoff with --detach:

    1. Download more patients   (data/pipeline_steps.py)
    2. Preprocess DICOM → numpy (data/pipeline_steps.py)
    3. Encode → MedVAE latents  (data/pipeline_steps.py)
    4. Train the model          (model/train_core.py)

All logic is imported from the shared modules — no duplication.

Run:   modal run --detach full_pipeline.py
Check: modal app logs ldct-full-pipeline
Get:   modal volume get ldct-data /results/diffusion_v2 ./results_v2
"""

import modal

app = modal.App("ldct-full-pipeline")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", "torchvision", "numpy", "matplotlib",
        "medvae", "pydicom", "tcia_utils", "requests", "pandas",
    )
    .add_local_python_source("data")
    .add_local_python_source("model")
)

vol = modal.Volume.from_name("ldct-data")

TARGET_PATIENTS = 50


@app.function(image=image, volumes={"/data": vol}, gpu="A100", timeout=50400) # 14 hours
def run_pipeline():
    import torch
    from data.pipeline_steps import (
        download_patients, preprocess_dicoms, encode_latents,
    )
    from model.train_core import run_training

    device = torch.device("cuda")

    # print("\n" + "=" * 70)
    # print("STEP 1/4: Download")
    # print("=" * 70)
    # download_patients(target_patients=TARGET_PATIENTS, vol=vol)

    # print("\n" + "=" * 70)
    # print("STEP 2/4: Preprocess")
    # print("=" * 70)
    # preprocess_dicoms(vol=vol)

    # print("\n" + "=" * 70)
    # print("STEP 3/4: Encode latents")
    # print("=" * 70)
    # encode_latents(device=device, vol=vol)

    print("\n" + "=" * 70)
    print("STEP 4/4: Train")
    print("=" * 70)
    run_training(
        latent_dir="/data/latents",
        results_dir="/data/results/diffusion_v2",
        vol=vol,
        num_epochs=80,
        device=device,
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("Download results: modal volume get ldct-data /results/diffusion_v2 ./results_v2")


@app.local_entrypoint()
def main():
    run_pipeline.remote()