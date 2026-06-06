"""
fid.py

FID (Frechet Inception Distance) for the pix2pix baseline.

FID compares the distribution of generated images to real ones in Inception
feature space. Lower is better. PSNR/SSIM measure per-pixel accuracy; FID
measures whether outputs look like real CT.

We report two numbers, mirroring the PSNR/SSIM eval:
    fid_pix2pix       - FID(full-dose, pix2pix output)
    fid_lowdose_input - FID(full-dose, low-dose input)   reference ("do nothing")
Improvement = lowdose - pix2pix.

Same test split (last 15% of patients) as eval_pix2pix.py, so the numbers line up.
This only reads the static pix2pix checkpoint and /data/processed, so it does not
touch the running flow job.

Note: FID is biased for small sample sizes, so trust the relative comparison
(pix2pix vs low-dose) more than the absolute number.

Run:  modal run eval/fid.py
Get:  modal volume get ldct-data /results/pix2pix_v2/eval/fid.json ./fid.json
"""

import modal
import os

app = modal.App("ldct-fid")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "torchmetrics[image]", "numpy")
    .add_local_python_source("baselines")
)

vol = modal.Volume.from_name("ldct-data")


def prep_for_fid(imgs):
    """(B, 1, H, W) in [-1, 1] -> (B, 3, 299, 299) float in [0, 1] for Inception.

    Reusable: the flow eval can call this on decoded images later.
    """
    import torch.nn.functional as F

    x = (imgs.clamp(-1, 1) + 1) / 2
    x = x.repeat(1, 3, 1, 1)
    return F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)


@app.function(image=image, volumes={"/data": vol}, gpu="A10G", timeout=1800)
def compute():
    import torch
    import numpy as np
    import json
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchmetrics.image.fid import FrechetInceptionDistance
    from baselines.pix2pix import UNetGenerator

    DATA_DIR = "/data/processed"
    RESULTS_DIR = "/data/results/pix2pix_v2"
    EVAL_DIR = f"{RESULTS_DIR}/eval"
    IMG_SIZE = 256
    FID_FEATURE = 2048  # Inception pool3 features (standard FID)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # Use the latest checkpoint (zero-padded names sort correctly).
    ckpt_dir = f"{RESULTS_DIR}/checkpoints"
    ckpts = sorted(f for f in os.listdir(ckpt_dir)
                   if f.startswith("epoch_") and f.endswith(".pt"))
    if not ckpts:
        raise FileNotFoundError(f"No epoch_*.pt in {ckpt_dir}")
    CHECKPOINT = os.path.join(ckpt_dir, ckpts[-1])

    device = torch.device("cuda")

    class TestPairs(Dataset):
        """Test-split (low_dose, full_dose) pairs in [-1, 1]. Same split as eval."""

        def __init__(self, root, img_size):
            self.img_size = img_size
            patients = sorted(
                d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
            )
            test_patients = patients[int(0.85 * len(patients)):]  # last 15%, by patient

            self.pairs = []
            for p in test_patients:
                ld_dir = os.path.join(root, p, "low_dose")
                fd_dir = os.path.join(root, p, "full_dose")
                if not (os.path.isdir(ld_dir) and os.path.isdir(fd_dir)):
                    continue
                ld_files = sorted(f for f in os.listdir(ld_dir) if f.endswith(".npy"))
                fd_files = sorted(f for f in os.listdir(fd_dir) if f.endswith(".npy"))
                for i in range(min(len(ld_files), len(fd_files))):
                    self.pairs.append((
                        os.path.join(ld_dir, ld_files[i]),
                        os.path.join(fd_dir, fd_files[i]),
                    ))
            print(f"FID test set: {len(test_patients)} patients, {len(self.pairs)} slices")

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            ld_path, fd_path = self.pairs[idx]
            low = torch.from_numpy(np.load(ld_path)).unsqueeze(0).float()
            full = torch.from_numpy(np.load(fd_path)).unsqueeze(0).float()
            if self.img_size != 512:
                low = F.interpolate(low.unsqueeze(0), size=self.img_size,
                                    mode="bilinear", align_corners=False).squeeze(0)
                full = F.interpolate(full.unsqueeze(0), size=self.img_size,
                                     mode="bilinear", align_corners=False).squeeze(0)
            return low * 2 - 1, full * 2 - 1

    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location=device)
    gen = UNetGenerator(in_ch=1, out_ch=1).to(device)
    gen.load_state_dict(ckpt["gen_state"])
    gen.eval()

    test_ds = TestPairs(DATA_DIR, IMG_SIZE)
    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

    fid_model = FrechetInceptionDistance(feature=FID_FEATURE, normalize=True).to(device)
    fid_input = FrechetInceptionDistance(feature=FID_FEATURE, normalize=True).to(device)

    print("Running inference + accumulating FID stats...")
    n = 0
    with torch.no_grad():
        for low, full in loader:
            low = low.to(device)
            full = full.to(device)
            fake = gen(low)

            real = prep_for_fid(full)
            fid_model.update(real, real=True)
            fid_model.update(prep_for_fid(fake), real=False)
            fid_input.update(real, real=True)
            fid_input.update(prep_for_fid(low), real=False)
            n += low.size(0)

    fid_pix2pix = float(fid_model.compute())
    fid_lowdose = float(fid_input.compute())

    print("\n" + "=" * 50)
    print("FID RESULTS  (lower is better)")
    print("=" * 50)
    print(f"  Low-dose input : {fid_lowdose:8.3f}")
    print(f"  Pix2Pix        : {fid_pix2pix:8.3f}")
    print(f"  Improvement    : {fid_lowdose - fid_pix2pix:+8.3f}")
    print(f"\n  Test slices: {n}  (FID is biased for small N)")

    summary = {
        "model": "pix2pix",
        "checkpoint": CHECKPOINT,
        "test_slices": n,
        "fid_feature": FID_FEATURE,
        "fid_pix2pix": round(fid_pix2pix, 3),
        "fid_lowdose_input": round(fid_lowdose, 3),
        "fid_improvement": round(fid_lowdose - fid_pix2pix, 3),
    }
    with open(f"{EVAL_DIR}/fid.json", "w") as f:
        json.dump(summary, f, indent=2)

    vol.commit()
    print(f"\nSaved to {EVAL_DIR}/fid.json")
    print("Download: modal volume get ldct-data /results/pix2pix_v2/eval/fid.json ./fid.json")


@app.local_entrypoint()
def main():
    compute.remote()
