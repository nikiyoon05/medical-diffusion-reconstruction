"""
vae.py

This is a wrapper to use Stanford MIMI's MedVAE that I found here: 
https://github.com/StanfordMIMI/MedVAE

This is for use in our latent diffusion pipeline. It was trained on x-ray images 
but should work well for our ct scan slices.

Loads a pretrained MedVAE model, freezes it, and gives clean encode/decode 
functions that work with our dataset tensors.

Encode/decode functions:
    https://github.com/StanfordMIMI/MedVAE/blob/main/documentation/inference.md

Usage:
also note that there are other models available (this one does 8x shrinking)
    vae = MedVAEWrapper(model_name="medvae_8_4_2d", device="cuda")

    # encode: image tensor to latent
    # input: (B, 1, 512, 512) float32, this is modified to [-1, 1]
    # output: (B, 4, 64, 64) float32
    latent = vae.encode(image_tensor)

    # decode: latent to image tensor
    # input: (B, 4, 64, 64) float32
    # output: (B, 1, 512, 512) full size image
    recon = vae.decode(latent)

To install: pip install medvae

"""

import torch
import torch.nn as nn
from medvae import MVAE
 
 
class MedVAEWrapper(nn.Module):
    """
    Frozen MedVAE wrapper for latent diffusion.
 
    args:
        model_name: Which medvae model to use. Recommended: "medvae_8_4_2d"
                    (8x spatial downsampling, 4 latent channels).
        modality:   "ct", "xray", or "mri". Controls input preprocessing.
        device:     "cuda" or "cpu".
    """
 
    # Expected latent shapes for each model (given 512x512 input)
    MODEL_INFO = {
        "medvae_4_1_2d": {"downsample": 4, "latent_channels": 1}, # shape of(B, 1, 128, 128)
        "medvae_4_3_2d": {"downsample": 4, "latent_channels": 3}, #shape(B, 3, 128, 128)
        "medvae_8_1_2d": {"downsample": 8, "latent_channels": 1},  # shape(B, 1, 64, 64)
        "medvae_8_4_2d": {"downsample": 8, "latent_channels": 4},  # shape(B, 4, 64, 64)
    }
 
    def __init__(self, model_name="medvae_8_4_2d", modality="ct", device="cuda"):
        super().__init__()
 
        if model_name not in self.MODEL_INFO:
            raise ValueError(
                f"unknown model: {model_name}. "
                f"Choose from: {list(self.MODEL_INFO.keys())}"
            )
 
        self.model_name = model_name
        self.info = self.MODEL_INFO[model_name]
        self.device = device
 
        # Load pretrained vae
        print(f"loading MedVAE: {model_name} (modality={modality})")
        self.model = MVAE(model_name=model_name, modality=modality).to(device)
 
        # freeze all weights — we never train the vae
        self.model.requires_grad_(False)
        self.model.eval()
 
        print(f"Spatial downsampling: {self.info['downsample']}x")
        print(f"Latent channels: {self.info['latent_channels']}")
        print(f"512x512 input → {512 // self.info['downsample']}x"
              f"{512 // self.info['downsample']}x{self.info['latent_channels']} latent")
 
    @torch.no_grad()
    def encode(self, x):
        """
        Encode image tensor to latent representation.
 
        args:
            x: (B, 1, H, W) float32 tensor, range [-1, 1]
 
        returns:
            latent: (B, C, H/d, W/d) float32 tensor
                    where C = latent_channels, d = downsample factor
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # (B, 1, H, W) --> (B, 3, H, W)
        return self.model.encode(x)
 
    @torch.no_grad()
    def decode(self, z):
        """
        Decode latent representation back to image.
 
        Args:
            z: (B, C, H_lat, W_lat) float32 tensor
 
        Returns:
            recon: (B, 1, H, W) float32 tensor
        """
        out = self.model.decode(z)
        if out.shape[1] == 3:
            out = out.mean(dim=1, keepdim=True)  # (B, 3, H, W) --> (B, 1, H, W)
        return out
 
    @torch.no_grad()
    def reconstruct(self, x):
        """
        Full encode → decode roundtrip. Useful for testing VAE quality.
 
        Args:
            x: (B, 1, H, W) float32 tensor, range [-1, 1]
 
        Returns:
            recon: (B, 1, H, W) float32 tensor
            latent: (B, C, H_lat, W_lat) float32 tensor
        """
        x_in = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
        decoded, latent = self.model(x_in, decode=True)
        if decoded.shape[1] == 3:
            decoded = decoded.mean(dim=1, keepdim=True)
        return decoded, latent
 
    @property
    def latent_channels(self):
        return self.info["latent_channels"]
 
    @property
    def downsample_factor(self):
        return self.info["downsample"]
 
    def latent_shape(self, img_size=512):
        """return the latent spatial dimensions for a given input size"""
        d = self.info["downsample"]
        c = self.info["latent_channels"]
        return (c, img_size // d, img_size // d)
 
 
# Quick test 
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = MedVAEWrapper(model_name="medvae_8_4_2d", device=device)
 
    # test with a dummy input
    dummy = torch.randn(1, 1, 512, 512).to(device)
    print(f"\nInput shape: {dummy.shape}")
 
    latent = vae.encode(dummy)
    print(f"Latent shape: {latent.shape}")
 
    recon = vae.decode(latent)
    print(f"Decoded shape: {recon.shape}")
 
    print(f"\nLatent info:{vae.latent_shape()}")
    print("VAE wrapper OK!!!!! yay")