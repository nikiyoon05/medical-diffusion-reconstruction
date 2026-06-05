"""
rectified_flow.py

This is rectified flow for condiitonal image-to-image translation.

THE IDEA!
Standard diffusion models learn complex, curved path from noise to data
This leanrs a stright line between them.

Following taken from lecture slides:
https://cs231n.stanford.edu/slides/2026/lecture_14.pdf

HOW IT WORKS (and steps)
1. Define endpoints:
    x0 = pure oise - N(0, I)
    x1 = full dose latent (the target)
2. Straight line interpolation at time t in [0, 1]:
    x_t = (1-t) * x0 + t * x1

3. Veloctiy alog this line:
    v = x1 - x0

4. train neural net to predict this velocity:
    v_pred = backbone(x_t, t, condition=low_dose_latent)

5. loss:
    L = MSE(v_pred, v)

CONDITIONAL GENERATION
in this task, the low dose latent is the condition. The backbone receives it as
additional input so it knows what to denoise. 
    Training: backbone(x_t, t, z_low_dose) --> pred = (z_full_dose - noise)
    inference: start from noise, condition on z_low_dose, step along predicted flow

INFERENCE
Start at t=0, with pure noise, step to t=1:
    z_0 = noise - N(0, I)
    for t in [0, dt, 2*dt, ..., 1-dt]
        v = model(z_t, t, z_low_dose)
        z_{t + dt} = z_t + dt * v # step
"""

import torch
import torch.nn as nn

class RectifiedFlow(nn.Module):
    """
    Rectified flow framework for conditional image to image

    This handles coputing and trianig loss, sampling and inference
    The model (ex U-Net or transformer) is passed in separately

    Args:
        model: nn.Module that takes(x_t, t, condition) --> predicted velocity
            Expected signature: model(x_t, t_emb, cond)
            where x_t: (B, C, H, W)
                  t_emb: (B, ) a float in [0, 1]
                  cond: (B, C, H, W)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def compute_loss(self, z_target, z_condition):
        """
        Compute recitified flow trainnig loss
        Args:
            z_target: (B, C, H, W) full dose latent (the target, x1)
            z_condition: (B, C, H, W) low-dose latent (the condition)
        Returns: 
            loss: scalalr, MSE between predicted and true velocity
        """
        B = z_target.shape[0]
        device = z_target.device

        # sample random noise, x0
        noise = torch.randn_like(z_target)

        # sample random timestep t, uniform from [0, 1]
        t = torch.rand(B, device=device)

        # linear combination along straight line
        t_expanded = t.view(B, 1, 1, 1) # need to brodcast to (B, C, H, W)
        x_t = (1 - t_expanded) * noise + t_expanded * z_target

        # true velocity 
        v_target = z_target - noise

        # predict velocity with model
        v_pred = self.model(x_t, t, z_condition)

        # mse loss
        loss = nn.functional.mse_loss(v_pred, v_target)

        return loss
    
    @torch.no_grad()
    def sample(self, z_condition, num_steps=30, device="cuda"):
        """
        Gerneate a full dose latent from a low-dose condition.
        Goes along learned flow:
            z_0 = oise
            z_{t + dt} = z_t + dt * model(z_t, t, condition)
            return z_1
        
        Args:
            z_condition: the low dose latent input (B, C, H, W)
            num_steps: number of steps
            device: torch device
        
        Returns:
            z_1: the predicted full dose latent (B, C, H, W)
        """
        # start from pure noise (z0)
        z_t = torch.randn_like(z_condition)

        # step size
        dt = 1.0/num_steps

        # now step
        for i in range(num_steps):
            # get t val
            t = torch.full((z_condition.shape[0],), i * dt, device=device)  # ensure broadcasting

            # predict velocity at current pred
            v_pred = self.model(z_t, t, z_condition)

            # step forward
            z_t = z_t + dt * v_pred
        
        return z_t
    
    @torch.no_grad()
    def sample_midpoint(self, z_condition, num_steps=30, device="cuda"):
        """
        Same as sample() but uses midpoint method for better accuracy.
        For each step [t, t+dt]:
            - Preict velocity at t: v1 = model(z_t, t, cond)
            - talk a half step: z_mid = z_t + (dt/2) * v1
            - predict velclity at midpoint v2 = model(z_mid, t + dt/2, cond)
            - take full step with midpoint velocity: z_{t + dt} = z_t + dt * v2
        """
        z_t = torch.randn_like(z_condition)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            # get t vals and midpoint vals
            t = torch.full((z_condition.shape[0],), i * dt, device=device)
            t_mid = torch.full((z_condition.shape[0],), (i + .5) * dt, device=device)

            # predict velocity at cur point
            v1 = self.model(z_t, t, z_condition)

            # half step to midpoint
            z_mid = z_t + (dt/2) * v1

            # predict velcity at midpoint
            v2 = self.model(z_mid, t_mid, z_condition)

            # full step using midpoint velocity
            z_t = z_t + dt * v2
        return z_t
    
    #-------------------SDE SAMPLER FOR VARIANCE--------------------
    @torch.no_grad()
    def sample_sde(self, z_condition, num_steps=50, sigma=0.1, device="cuda"):
        """
        Stochastic sampling by converting the ODE to an SDE.
 
        Instead of:  z_{t+dt} = z_t + dt * v                     (ODE, deterministic)
        We do:       z_{t+dt} = z_t + dt * v + sigma*sqrt(dt)*ε   (SDE, stochastic)
 
        The injected noise makes each run produce a slightly different output.
        Areas where the model is confident will be stable across runs.
        Areas where it's uncertain will vary — captured by pixel variance.
 
        Args:
            z_condition: (B, C, H, W) low-dose latent
            num_steps:  number of steps (more = smoother)
            sigma:  noise strength — controls randomness level
                    0.0 = deterministic (same as ODE)
                    0.05-0.2 = typical range for uncertainty estimation
            device: torch device
 
        Returns:
            z_1: (B, C, H, W) predicted full-dose latent (stochastic)
        """
        z_t = torch.randn_like(z_condition)
        dt = 1.0 / num_steps
        sqrt_dt = dt ** 0.5
 
        for i in range(num_steps):
            t = torch.full((z_condition.shape[0],), i * dt, device=device)
 
            # Predicted velocity (deterministic part)
            v_pred = self.backbone(z_t, t, z_condition)
 
            # Langevin noise (stochastic part)
            noise = torch.randn_like(z_t)
 
            # SDE step: drift + diffusion
            z_t = z_t + dt * v_pred + sigma * sqrt_dt * noise
 
        return z_t
    
    @torch.no_grad()
    def compute_uncertainty(self, z_condition, n_runs=10, num_steps=50,
                            sigma=0.1, device="cuda"):
        """
        Estimate per-pixel uncertainty via Monte Carlo SDE sampling.
 
        Runs the SDE sampler n_runs times on the same input, then computes:
            - mean prediction (best estimate of the full-dose image)
            - variance map (per-pixel uncertainty)
 
        High variance = the model needed the stochastic noise to "decide"
        what to put there = low confidence = potential hallucination.
 
        Args:
            z_condition: (B, C, H, W) low-dose latent
            n_runs: number of stochastic runs (10 is typical)
            num_steps: ODE/SDE steps per run
            sigma: noise strength for SDE
            device:
 
        Returns:
            mean_latent:(B, C, H, W) averaged prediction across runs
            variance_latent: (B, C, H, W) per-element variance across runs
            all_samples: list of n_runs tensors, each (B, C, H, W)
        """
        all_samples = []
 
        for run in range(n_runs):
            z_pred = self.sample_sde(
                z_condition, num_steps=num_steps, sigma=sigma, device=device
            )
            all_samples.append(z_pred)
 
        # Stack: (n_runs, B, C, H, W)
        stacked = torch.stack(all_samples, dim=0)
 
        # Mean and variance across the n_runs dimension
        mean_latent = stacked.mean(dim=0)       # (B, C, H, W)
        variance_latent = stacked.var(dim=0)     # (B, C, H, W)
 
        return mean_latent, variance_latent, all_samples

    # -------------------------------------------------------------
    # ----------- just a quick test --------------------
    if __name__ == "main":
        # Dummy backbone for testing
        class DummyBackbone(nn.Module):
            """Placeholder — just returns zeros. Real backbone goes in dit.py."""
            def __init__(self, channels=4):
                super().__init__()
                self.net = nn.Conv2d(channels * 2, channels, 1)  # x_t + cond → v
    
            def forward(self, x_t, t, cond):
                # Real backbone will also use t for timestep embedding
                x = torch.cat([x_t, cond], dim=1)
                return self.net(x)
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
        backbone = DummyBackbone(channels=4).to(device)
        flow = RectifiedFlow(backbone).to(device)
    
        # simulate MedVAE latents: (B, 4, 64, 64)
        B = 2
        z_full = torch.randn(B, 4, 64, 64).to(device)   # full-dose latent
        z_low = torch.randn(B, 4, 64, 64).to(device)     # low-dose latent
    
        # Test training loss
        loss = flow.compute_loss(z_target=z_full, z_condition=z_low)
        print(f"Training loss: {loss.item():.4f}")
    
        # test sampling
        z_pred = flow.sample(z_low, num_steps=5, device=device)
        print(f"Sampled shape: {z_pred.shape}")
    
        # Test sampling (midpoint)
        z_pred2 = flow.sample_midpoint(z_low, num_steps=5, device=device)
        print(f"Midpoint shape: {z_pred2.shape}")
    
        print("\nRectified flow OK!")