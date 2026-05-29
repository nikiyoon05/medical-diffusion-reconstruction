import modal

app = modal.App()

image = modal.Image.debian_slim().pip_install(
    "torch",
    "torchvision",
    "diffusers",
    "pydicom",
    "numpy",
    "Pillow",
    "tqdm",
    "accelerate",
)

@app.function(gpu="A10G", image=image)
def test():
    import torch
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

@app.local_entrypoint()
def main():
    test.remote()