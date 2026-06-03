# Implementation Plan: Latent Rectified Flow for Medical Image Reconstruction

> Based on CS 231N Milestone 2 Technical Approach (Yoon, Burgett, Rhee).
> Goal: reconstruct Normal-Dose CT from Low-Dose CT via a latent-space
> rectified flow, with a U-Net baseline and per-pixel uncertainty maps.
> Hard constraint: end-to-end train + eval under a **$100 compute budget** on a single GPU.

## 0. Guiding Principles
- **Paired data, straight paths.** We have aligned Low-Dose (LD) / Normal-Dose (ND)
  slices, so we learn a straight-line vector field LD→ND instead of diffusing to Gaussian noise.
- **Latent + flow = feasible.** Latents cut dimensionality ~90%; straight flows cut
  ODE steps to 5–10. Both are required to fit the budget.
- **Ship the baseline first.** A Residual U-Net is the comparison point and a sanity check
  on the data pipeline before any flow code is written.

---

## 1. Project Scaffolding
- [ ] Create package layout:
  ```
  src/
    data/        # dataset, slicing, paired loaders, normalization
    models/
      vae.py     # frozen pretrained VAE wrapper (encode/decode)
      unet.py    # residual U-Net baseline
      flow.py    # velocity-field network (U-Net or DiT-style transformer)
    training/
      train_baseline.py
      train_flow.py
      losses.py
    sampling/
      ode_solver.py   # deterministic Euler/Heun integrator
      sde_solver.py   # Option A: Langevin-noise SDE sampler
      likelihood.py   # Option B: Hutchinson trace / CNF log-density
    eval/
      metrics.py      # SSIM, FID
      uncertainty.py  # variance heatmaps, log p(x) maps
    utils/            # config, logging, checkpoint, seeding, AMP helpers
  configs/            # YAML configs (baseline.yaml, flow.yaml)
  scripts/            # download/prepare data, run experiments
  notebooks/          # quick visual inspection
  ```
- [ ] `requirements.txt` (pin versions): `torch`, `torchvision`, `diffusers`
  (for the pretrained VAE), `numpy`, `pydicom`/`nibabel` (CT I/O),
  `scikit-image` (SSIM), `pytorch-fid` or `torchmetrics[image]` (FID),
  `einops`, `pyyaml`, `tqdm`, `wandb` (optional).
- [ ] Config + seeding + AMP/mixed-precision scaffolding in `utils/`.

## 2. Data Pipeline — Mayo Clinic Low-Dose CT Grand Challenge
- [ ] Download script for the Mayo LDCT dataset (DICOM volumes).
- [ ] Slice 3D volumes → 2D axial slices; persist paired (LD, ND) slices.
- [ ] Preprocessing: HU windowing, resize/center to 512×512, normalize to the
  VAE's expected input range (typically `[-1, 1]`).
- [ ] `PairedSliceDataset`: returns `(ld_slice, nd_slice)` exactly registered.
- [ ] Train/val/test split **by patient** (avoid slice leakage across splits).
- [ ] Smoke test: visualize a few LD/ND pairs to confirm alignment.

## 3. Baseline — Residual U-Net (do this first)
- [ ] `models/unet.py`: residual U-Net, direct pixel-space LD→ND mapping (no latent, no flow).
- [ ] `training/train_baseline.py`: L1/MSE loss, AdamW + cosine LR, AMP.
- [ ] Log SSIM on val each epoch; checkpoint best.
- [ ] This validates the data pipeline and gives the comparison baseline.

## 4. VAE (frozen)
- [ ] `models/vae.py`: load a pretrained VAE (e.g. Stable Diffusion `AutoencoderKL`
  from `diffusers`), **freeze** all weights.
- [ ] `encode(x) -> z` (scaled by the VAE's latent scaling factor) and `decode(z) -> x`.
- [ ] Validation step: encode→decode an ND slice and confirm reconstruction quality
  is acceptable on CT (medical images differ from natural images — verify before relying on it).
  - Risk note: if the natural-image VAE reconstructs CT poorly, plan B is light VAE
    fine-tuning on CT slices (budget permitting) or a smaller domain VAE.

## 5. Rectified Flow Core
- [ ] `models/flow.py`: velocity field `v_θ(Z_t, t)` operating in latent space.
  Start with a time-conditioned U-Net; optionally a small DiT-style transformer.
- [ ] `training/losses.py` — velocity matching:
  - Sample paired (LD, ND); encode to `Z₀, Z₁`.
  - Sample `t ∈ [0,1]`; interpolate `Z_t = t·Z₁ + (1−t)·Z₀`.
  - Loss: `L_flow = || v_θ(Z_t, t) − (Z₁ − Z₀) ||²`.
- [ ] `training/train_flow.py`: AdamW, cosine LR, mixed precision, gradient clipping,
  EMA of weights (recommended for flow models), checkpointing.

## 6. Inference / Sampling
- [ ] `sampling/ode_solver.py`: deterministic Euler (and Heun) integrator from
  `Z₀` to `Z₁` in **5–10 steps**; decode `Z₁` via the VAE.
- [ ] End-to-end inference: LD slice → encode → ODE integrate → decode → ND estimate.
- [ ] Sweep step counts {5, 8, 10, 20} and record SSIM/FID vs. steps (the "fewer steps" claim).

## 7. Evaluation
- [ ] `eval/metrics.py`: SSIM (structure preserved) and FID (texture realism).
- [ ] Comparison table: Residual U-Net vs. Latent Rectified Flow on test set.
- [ ] Qualitative panels: LD input | U-Net | Flow | ND ground truth.

## 8. Uncertainty / The Clinical Edge
Both paths reuse the trained flow model — no retraining needed.
- [ ] **Option A — SDE Variance (Monte Carlo).** `sampling/sde_solver.py`:
  convert the ODE to an SDE by injecting Langevin noise; run the same slice ~10×
  → pixel-wise variance → **variance heatmap** flagging low-confidence regions.
- [ ] **Option B — Exact Log-Likelihood (CNF).** `sampling/likelihood.py`:
  treat the rectified flow as a Continuous Normalizing Flow; use the **Hutchinson
  trace estimator** to compute the change in log-density from `Z₀` to `Z₁` → a
  formal `log p(x)` score.
- [ ] `eval/uncertainty.py`: produce reconstruction + confidence map; brief clinical
  comparison of A vs. B (which is more useful/interpretable).

## 9. Budget & Compute Discipline (<$100)
- [ ] Track GPU-hours; prototype on a small slice subset before full runs.
- [ ] Use AMP, latent compression, and small step counts to stay cheap.
- [ ] Cache encoded latents for the training set to avoid re-encoding every epoch
  (big speedup since the VAE is frozen).
- [ ] Log estimated cost per run; stop-loss if approaching the cap.

## 10. Suggested Order of Execution
1. Scaffolding + data pipeline (Sections 1–2)
2. Residual U-Net baseline (Section 3) — proves data + gives comparison
3. Frozen VAE + encode/decode sanity check (Section 4)
4. Rectified flow training (Section 5)
5. ODE sampling + evaluation vs. baseline (Sections 6–7)
6. Uncertainty Option A, then Option B (Section 8)
7. Final comparison tables, figures, cost accounting (Sections 7, 9)

## Open Questions / Decisions to Confirm
- Which pretrained VAE, and does it reconstruct CT acceptably without fine-tuning?
- Latent-space velocity net: U-Net vs. transformer (start U-Net)?
- Target slice resolution / latent shape, and final latent normalization.
- Target GPU (Colab/Lambda/etc.) and the cost-per-hour assumption behind the $100 cap.
