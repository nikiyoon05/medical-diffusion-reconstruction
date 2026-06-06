"""
likelihood.py  (Option B — exact log-likelihood / Continuous Normalizing Flow)

Confidence scoring via a formal log p(x) score, as outlined in the milestone slides.
This is the deterministic-ODE counterpart to the SDE variance heatmaps (Option A).

SELF-CONTAINED ON PURPOSE:
    - Does NOT modify any existing code. The CNF log-likelihood + a matching Euler
      sampler are defined as standalone functions in this file. It only *reads* the
      existing model (ConditionalUNet) and VAE wrapper to load weights / decode.
    - Loads latents directly from the volume (the `data` dataset package is not in
      the repo), so nothing else is needed.
    - Writes to a brand-new results folder so existing results are untouched.

THE MATH
Because our rectified flow is an ODE  dz/dt = v(z, t, cond), it is a Continuous
Normalizing Flow. The instantaneous change-of-variables formula gives:

    d/dt log p(z_t) = -tr(dv/dz)
    =>  log p(z_1 | cond) = log p_0(z_0) - integral_0^1 tr(dv/dz) dt

We recover z_0 by integrating the ODE backward from a data latent and accumulate
the trace integral along the trajectory. The trace is estimated with the Hutchinson
estimator (one vector-Jacobian product per probe instead of D = 4*64*64 = 16384):

    tr(dv/dz) ≈ E_eps[ eps^T (dv/dz) eps ],   eps ~ N(0, I)

Higher log p  =>  the latent looks more like an in-distribution (full-dose) scan
=>  higher confidence in the reconstruction.

For each held-out slice we score three latents under the model:
    - z_full  (real full-dose GT)  -> should be HIGH likelihood
    - z_pred  (our reconstruction) -> high if the model is confident
    - z_low   (low-dose input)     -> LOW likelihood (noisy, out-of-distribution)

NOTE on the split: the original train/val/test split lived in the uncommitted
`data.latent_dataset` module, so we score a deterministic held-out tail of patients
(by sorted ID). The qualitative log p(x) story is robust to the exact split.

Run: modal run eval/likelihood.py
Get: modal volume get ldct-data /results/likelihood_optionB ./likelihood_optionB
"""

import modal

app = modal.App("ldct-likelihood")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "matplotlib", "medvae")
    .add_local_python_source("model")
)

vol = modal.Volume.from_name("ldct-data")

# Config — current main-branch model is the cross-attention U-Net => diffusion_v3
CHECKPOINT = "/data/results/diffusion_v3/checkpoints/best.pt"
LATENT_ROOT = "/data/latents"
RESULTS_OUT = "/data/results/likelihood_optionB"   # NEW folder; nothing existing touched
NUM_STEPS = 50          # ODE steps for sampling and the likelihood integration
N_HUTCHINSON = 1        # trace probes per step (1 is standard FFJORD; raise to cut variance)
N_SLICES = 64           # how many held-out slices to score
BATCH_SIZE = 8
N_EXAMPLES = 8          # annotated example panels to render
EVAL_PATIENT_FRAC = 0.15  # held-out tail of patients (by sorted ID) used for eval
SEED = 0


# ----------------------------------------------------------------------
# Standalone CNF helpers (defined here so no existing file is modified)
# ----------------------------------------------------------------------
def ode_sample(model, z_condition, num_steps, device):
    """Deterministic Euler integration noise -> data (mirrors the trained sampler)."""
    import torch
    with torch.no_grad():
        z_t = torch.randn_like(z_condition)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((z_condition.shape[0],), i * dt, device=device)
            z_t = z_t + dt * model(z_t, t, z_condition)
    return z_t


def cnf_log_likelihood(model, z_data, z_condition, num_steps=50,
                       n_hutchinson=1, device="cuda"):
    """
    log p(z_data | z_condition) under the rectified flow treated as a CNF.

    Integrates the ODE backward from the data latent to the base Gaussian and
    accumulates integral_0^1 tr(dv/dz) dt with the Hutchinson estimator.

    Returns:
        log_px:   (B,) log-density in nats (higher = more likely)
        bits_dim: (B,) normalized negative log-likelihood, bits/dim (lower = more likely)
        z0:       (B, C, H, W) recovered base point
    """
    import math
    import torch

    B = z_data.shape[0]
    D = z_data[0].numel()
    dt = 1.0 / num_steps

    z = z_data.detach()
    delta_logp = torch.zeros(B, device=device)  # accumulates integral_0^1 tr dt

    for i in range(num_steps):
        t_val = 1.0 - i * dt
        t = torch.full((B,), t_val, device=device)

        with torch.enable_grad():
            z = z.detach().requires_grad_(True)
            v = model(z, t, z_condition)

            trace = torch.zeros(B, device=device)
            for j in range(n_hutchinson):
                eps = torch.randn_like(z)
                (vjp,) = torch.autograd.grad(
                    v, z, grad_outputs=eps,
                    retain_graph=(j < n_hutchinson - 1),
                )
                trace = trace + (vjp * eps).flatten(1).sum(dim=1)
            trace = trace / n_hutchinson

        delta_logp = delta_logp + trace.detach() * dt          # left-Riemann sum
        z = (z - dt * v.detach()).detach()                     # backward Euler step

    z0 = z
    log_p0 = -0.5 * (z0.flatten(1).pow(2).sum(dim=1) + D * math.log(2 * math.pi))
    log_px = log_p0 - delta_logp
    bits_dim = -log_px / (D * math.log(2))
    return log_px, bits_dim, z0


@app.function(image=image, volumes={"/data": vol}, gpu="A100", timeout=3600)
def evaluate_likelihood():
    import os
    import json
    import math
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from model.unet import ConditionalUNet
    from model.vae import MedVAEWrapper

    device = torch.device("cuda")
    os.makedirs(f"{RESULTS_OUT}/examples", exist_ok=True)

    # ------------------------------------------------------------------
    # 0. Math self-test: closed-form linear flow v(z,t,cond)=c*z.
    #    z0 = z1*exp(-c), tr(dv/dz)=c*D, integral_0^1 tr dt = c*D, so
    #    log p(z1) = -0.5*(exp(-2c)||z1||^2 + D ln(2pi)) - c*D   (analytic)
    # ------------------------------------------------------------------
    class LinearVelocity(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c = c

        def forward(self, x_t, t, cond):
            return self.c * x_t

    print("Running math self-test (linear flow vs. analytic)...")
    c = -0.5
    z1 = torch.randn(4, 4, 64, 64, device=device)
    cond0 = torch.zeros_like(z1)
    lp, _, _ = cnf_log_likelihood(LinearVelocity(c), z1, cond0,
                                  num_steps=200, n_hutchinson=8, device=device)
    D = z1[0].numel()
    analytic = (-0.5 * (math.exp(-2 * c) * z1.flatten(1).pow(2).sum(1)
                        + D * math.log(2 * math.pi)) - c * D)
    rel_err = (lp - analytic).abs() / analytic.abs()
    print(f"  method   = {[round(x, 1) for x in lp.tolist()]}")
    print(f"  analytic = {[round(x, 1) for x in analytic.tolist()]}")
    print(f"  max rel err = {rel_err.max().item():.4%}")
    assert rel_err.max().item() < 0.01, "log_likelihood self-test FAILED (>1% error)"
    print("  self-test PASSED\n")

    # ------------------------------------------------------------------
    # Image-space SSIM (for the quality-vs-likelihood correlation)
    # ------------------------------------------------------------------
    def ssim(pred, target, window=11):
        C1, C2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2
        sigma = 1.5
        coords = torch.arange(window, dtype=torch.float32) - window // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g = g / g.sum()
        k = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0).to(pred.device)
        mu1 = torch.nn.functional.conv2d(pred, k, padding=window // 2)
        mu2 = torch.nn.functional.conv2d(target, k, padding=window // 2)
        m1, m2, m12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        s1 = torch.nn.functional.conv2d(pred ** 2, k, padding=window // 2) - m1
        s2 = torch.nn.functional.conv2d(target ** 2, k, padding=window // 2) - m2
        s12 = torch.nn.functional.conv2d(pred * target, k, padding=window // 2) - m12
        smap = ((2 * m12 + C1) * (2 * s12 + C2)) / ((m1 + m2 + C1) * (s1 + s2 + C2))
        return smap.mean().item()

    # ------------------------------------------------------------------
    # 1. Held-out eval items, loaded straight from the volume
    # ------------------------------------------------------------------
    patients = sorted(p for p in os.listdir(LATENT_ROOT) if p.startswith("C"))
    n_eval_p = max(1, int(round(len(patients) * EVAL_PATIENT_FRAC)))
    eval_patients = patients[-n_eval_p:]
    print(f"Eval patients (held-out tail): {eval_patients}")

    items = []  # (low_path, full_path)
    for p in eval_patients:
        low_dir = f"{LATENT_ROOT}/{p}/low_dose"
        for fn in sorted(os.listdir(low_dir)):
            full_path = f"{LATENT_ROOT}/{p}/full_dose/{fn}"
            if os.path.exists(full_path):
                items.append((f"{low_dir}/{fn}", full_path))

    rng = np.random.default_rng(SEED)
    if len(items) > N_SLICES:
        sel = sorted(rng.choice(len(items), size=N_SLICES, replace=False).tolist())
        items = [items[i] for i in sel]
    print(f"Scoring {len(items)} slices from {len(eval_patients)} held-out patients\n")

    def load_batch(batch_items):
        lows = np.stack([np.load(lp) for lp, _ in batch_items])
        fulls = np.stack([np.load(fp) for _, fp in batch_items])
        return (torch.from_numpy(lows).float().to(device),
                torch.from_numpy(fulls).float().to(device))

    # ------------------------------------------------------------------
    # 2. Model (EMA weights). Arch matches training (main / train_core.py).
    # ------------------------------------------------------------------
    print("Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT, map_location=device)
    print(f"  Epoch {ckpt.get('epoch', '?')}, val_loss {ckpt.get('val_loss', '?')}")

    backbone = ConditionalUNet(
        in_channels=4, cond_channels=4, base_channels=128,
        channel_mults=(1, 2, 4, 4), num_res_blocks=3,
        dropout=0.0, attn_resolutions=(8, 16),
        cross_attn_resolutions=(8, 16), n_heads=4,
    ).to(device)
    backbone.load_state_dict(ckpt["backbone_state"])  # EMA weights
    backbone.eval()
    for prm in backbone.parameters():   # only need grads w.r.t. z, not params
        prm.requires_grad_(False)

    vae = MedVAEWrapper(model_name="medvae_8_4_2d", device=device)

    # ------------------------------------------------------------------
    # 3. Score each latent type
    # ------------------------------------------------------------------
    print(f"Scoring ({NUM_STEPS} ODE steps, {N_HUTCHINSON} Hutchinson probe(s))...")
    logp_full, logp_pred, logp_low = [], [], []
    bpd_full, bpd_pred, bpd_low = [], [], []
    pred_ssim = []

    for bi in range(0, len(items), BATCH_SIZE):
        z_low, z_full = load_batch(items[bi:bi + BATCH_SIZE])
        z_pred = ode_sample(backbone, z_low, NUM_STEPS, device)

        lp_f, bd_f, _ = cnf_log_likelihood(backbone, z_full, z_low, NUM_STEPS, N_HUTCHINSON, device)
        lp_p, bd_p, _ = cnf_log_likelihood(backbone, z_pred, z_low, NUM_STEPS, N_HUTCHINSON, device)
        lp_l, bd_l, _ = cnf_log_likelihood(backbone, z_low,  z_low, NUM_STEPS, N_HUTCHINSON, device)

        logp_full += lp_f.tolist();  bpd_full += bd_f.tolist()
        logp_pred += lp_p.tolist();  bpd_pred += bd_p.tolist()
        logp_low  += lp_l.tolist();  bpd_low  += bd_l.tolist()

        with torch.no_grad():
            pred_imgs = vae.decode(z_pred)
            full_imgs = vae.decode(z_full)
            for i in range(pred_imgs.size(0)):
                pred_ssim.append(ssim(pred_imgs[i:i+1], full_imgs[i:i+1]))

        print(f"  {min(bi + BATCH_SIZE, len(items))}/{len(items)} slices "
              f"| logp[full/pred/low] = {np.mean(lp_f.tolist()):.0f} / "
              f"{np.mean(lp_p.tolist()):.0f} / {np.mean(lp_l.tolist()):.0f}")

    logp_full = np.array(logp_full); logp_pred = np.array(logp_pred); logp_low = np.array(logp_low)
    pred_ssim = np.array(pred_ssim)

    def stats(a):
        return {"mean": round(float(np.mean(a)), 2), "std": round(float(np.std(a)), 2)}

    corr = float(np.corrcoef(logp_pred, pred_ssim)[0, 1])

    print("\n" + "=" * 64)
    print("OPTION B — EXACT LOG-LIKELIHOOD (nats; higher = more in-distribution)")
    print("=" * 64)
    print(f"  {'Full-dose (GT)':22s} log p = {np.mean(logp_full):12.1f} +/- {np.std(logp_full):.1f}")
    print(f"  {'Reconstruction (ours)':22s} log p = {np.mean(logp_pred):12.1f} +/- {np.std(logp_pred):.1f}")
    print(f"  {'Low-dose (input)':22s} log p = {np.mean(logp_low):12.1f} +/- {np.std(logp_low):.1f}")
    print(f"\n  corr(log p(recon), SSIM) = {corr:+.3f}")

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bins = 30
    ax.hist(logp_full, bins=bins, alpha=0.6, label="Full-dose (GT)", color="tab:green")
    ax.hist(logp_pred, bins=bins, alpha=0.6, label="Reconstruction (ours)", color="tab:blue")
    ax.hist(logp_low,  bins=bins, alpha=0.6, label="Low-dose (input)", color="tab:red")
    ax.set_xlabel("log p(x)  (nats)")
    ax.set_ylabel("count")
    ax.set_title("CNF log-likelihood by latent type\n(higher = more like a real full-dose scan)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_OUT}/logp_histogram.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.scatter(logp_pred, pred_ssim, alpha=0.5, s=12, color="tab:blue")
    ax.set_xlabel("log p(reconstruction)  (nats)")
    ax.set_ylabel("SSIM(reconstruction, GT)")
    ax.set_title(f"Does likelihood predict reconstruction quality?\nPearson r = {corr:.3f}")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_OUT}/logp_vs_ssim.png", dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # 5. Annotated example panels (low-dose | recon w/ log p | GT w/ log p)
    # ------------------------------------------------------------------
    print("\nRendering annotated example panels...")
    example_items = items[:: max(1, len(items) // N_EXAMPLES)][:N_EXAMPLES]
    for k, it in enumerate(example_items):
        z_low, z_full = load_batch([it])
        z_pred = ode_sample(backbone, z_low, NUM_STEPS, device)
        lp_p, _, _ = cnf_log_likelihood(backbone, z_pred, z_low, NUM_STEPS, N_HUTCHINSON, device)
        lp_f, _, _ = cnf_log_likelihood(backbone, z_full, z_low, NUM_STEPS, N_HUTCHINSON, device)

        with torch.no_grad():
            low_np = vae.decode(z_low)[0, 0].cpu().numpy()
            pred_np = vae.decode(z_pred)[0, 0].cpu().numpy()
            full_np = vae.decode(z_full)[0, 0].cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(low_np, cmap="gray", vmin=-1, vmax=1)
        axes[0].set_title("Low-dose input"); axes[0].axis("off")
        axes[1].imshow(pred_np, cmap="gray", vmin=-1, vmax=1)
        axes[1].set_title(f"Reconstruction\nlog p = {lp_p.item():.0f} nats"); axes[1].axis("off")
        axes[2].imshow(full_np, cmap="gray", vmin=-1, vmax=1)
        axes[2].set_title(f"Full-dose (GT)\nlog p = {lp_f.item():.0f} nats"); axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(f"{RESULTS_OUT}/examples/example_{k:02d}.png", dpi=130, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # 6. Save metrics
    # ------------------------------------------------------------------
    summary = {
        "checkpoint": CHECKPOINT,
        "best_epoch": ckpt.get("epoch"),
        "num_steps": NUM_STEPS,
        "n_hutchinson": N_HUTCHINSON,
        "n_slices": len(items),
        "eval_patients": eval_patients,
        "logp_nats": {
            "full_dose_gt": stats(logp_full),
            "reconstruction": stats(logp_pred),
            "low_dose_input": stats(logp_low),
        },
        "bits_per_dim": {
            "full_dose_gt": stats(np.array(bpd_full)),
            "reconstruction": stats(np.array(bpd_pred)),
            "low_dose_input": stats(np.array(bpd_low)),
        },
        "corr_logp_recon_vs_ssim": round(corr, 3),
    }
    with open(f"{RESULTS_OUT}/likelihood_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    vol.commit()
    print(f"\nSaved to {RESULTS_OUT}")
    print("Download: modal volume get ldct-data /results/likelihood_optionB ./likelihood_optionB")


@app.local_entrypoint()
def main():
    evaluate_likelihood.remote()
