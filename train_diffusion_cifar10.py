"""Train the Assignment 4 diffusion model (U-Net, DDPM noise prediction) on CIFAR-10.

Usage:  uv run --with matplotlib python train_diffusion_cifar10.py            (default 30 epochs)
        uv run --with matplotlib python train_diffusion_cifar10.py --epochs 50
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from helper_lib.data_loader import get_cifar10_generation_loader
from helper_lib.generator import generate_diffusion_samples, save_samples
from helper_lib.model import DiffusionModel, UNet
from helper_lib.trainer import train_diffusion
from helper_lib.utils import get_device

MODEL_PATH = Path("models/diffusion_unet.pth")
METRICS_PATH = Path("models/diffusion_metrics.json")
FIG_DIR = Path("docs/screenshots")
SNAPSHOT_EPOCHS = {1, 5, 10, 20}  # plus the final epoch
UNET_CONFIG = dict(image_size=32, in_channels=3, widths=(32, 64, 96, 128), block_depth=2, embedding_dim=32)


def plot_loss(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping loss plot")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(range(1, len(history["loss"]) + 1), history["loss"], marker="o", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (predicted vs. true noise)")
    ax.set_title("Diffusion model training loss")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    loader = get_cifar10_generation_loader(args.data_dir, batch_size=args.batch_size)
    unet = UNet(**UNET_CONFIG)
    diffusion = DiffusionModel(unet, timesteps=1000)
    print(f"U-Net parameters: {sum(p.numel() for p in unet.parameters()):,}")

    start = time.time()

    def snapshot(epoch, ema_unet):
        print(f"  elapsed: {(time.time() - start) / 60:.1f} min")
        if epoch in SNAPSHOT_EPOCHS or epoch == args.epochs:
            sampler = DiffusionModel(ema_unet, timesteps=1000).to(device)
            images = generate_diffusion_samples(sampler, num_samples=64, steps=50, device=device, seed=args.seed)
            save_samples(images, FIG_DIR / f"diffusion_epoch_{epoch:02d}.png")

    criterion = nn.MSELoss()  # DDPM objective: MSE between predicted and true noise
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-4)
    diffusion = train_diffusion(diffusion, loader, criterion, optimizer, device=device, epochs=args.epochs,
                                on_epoch_end=snapshot)
    ema_unet, history = diffusion.ema_unet.to("cpu"), diffusion.history
    torch.save({"state_dict": ema_unet.state_dict(), "config": UNET_CONFIG, "timesteps": 1000,
                "epochs": args.epochs}, MODEL_PATH)

    final = DiffusionModel(ema_unet, timesteps=1000)
    save_samples(generate_diffusion_samples(final, num_samples=64, steps=50, seed=7), FIG_DIR / "diffusion_samples_final.png")
    plot_loss(history, FIG_DIR / "diffusion_training_loss.png")

    METRICS_PATH.write_text(json.dumps({
        "model": "Diffusion model (U-Net noise predictor, DDPM objective, cosine schedule)",
        "dataset": "CIFAR-10", "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.lr, "optimizer": "AdamW (weight decay 1e-4)", "ema_decay": 0.999,
        "timesteps": 1000, "sampler": "DDIM, 50 steps", "unet_config": UNET_CONFIG,
        "unet_parameters": sum(p.numel() for p in ema_unet.parameters()),
        "final_loss": round(history["loss"][-1], 4),
        "training_minutes": round((time.time() - start) / 60, 1),
        "history": {"loss": [round(v, 4) for v in history["loss"]]},
    }, indent=2))
    print(f"Saved {MODEL_PATH}, {METRICS_PATH} and sample grids in {FIG_DIR}/")



if __name__ == "__main__":
    main()
