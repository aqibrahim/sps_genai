"""Train the Assignment 3 GAN on MNIST and save it for the API.

Usage:  uv run --with matplotlib python train_gan_mnist.py            (default 30 epochs)
        uv run --with matplotlib python train_gan_mnist.py --epochs 50
(matplotlib is optional; it is only used to plot the training curves.)
"""
import argparse
import json
from pathlib import Path

import torch

from helper_lib.data_loader import get_mnist_loader
from helper_lib.generator import generate_gan_samples, save_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.utils import get_device

GEN_PATH = Path("models/gan_generator.pth")
DISC_PATH = Path("models/gan_discriminator.pth")
METRICS_PATH = Path("models/gan_metrics.json")
FIG_DIR = Path("docs/screenshots")
SNAPSHOT_EPOCHS = {1, 5, 10, 20}  # plus the final epoch


def plot_curves(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping training-curve plot")
        return
    epochs = range(1, len(history["d_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["d_loss"], label="Discriminator loss")
    ax1.plot(epochs, history["g_loss"], label="Generator loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE loss"); ax1.set_title("GAN losses"); ax1.legend()
    ax2.plot(epochs, history["d_real"], label="D(x): real images")
    ax2.plot(epochs, history["d_fake"], label="D(G(z)): generated images")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="0.5 (can't tell apart)")
    ax2.set_ylim(0, 1); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Mean discriminator output")
    ax2.set_title("Discriminator outputs"); ax2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    GEN_PATH.parent.mkdir(exist_ok=True)

    loader = get_mnist_loader(args.data_dir, batch_size=args.batch_size)
    gan = get_model("GAN")

    def snapshot(epoch, generator):
        # Same fixed noise every time, so the grids show how the same 64 samples evolve
        if epoch in SNAPSHOT_EPOCHS or epoch == args.epochs:
            images = generate_gan_samples(generator, num_samples=64, device=device, seed=args.seed)
            save_samples(images, FIG_DIR / f"gan_epoch_{epoch:02d}.png")

    history = train_gan(gan, loader, device=device, epochs=args.epochs, lr=args.lr, on_epoch_end=snapshot)

    G, D = gan.generator.to("cpu"), gan.discriminator.to("cpu")
    torch.save({"state_dict": G.state_dict(), "latent_dim": G.latent_dim, "epochs": args.epochs}, GEN_PATH)
    torch.save({"state_dict": D.state_dict(), "epochs": args.epochs}, DISC_PATH)

    save_samples(generate_gan_samples(G, num_samples=64, seed=7), FIG_DIR / "gan_samples_final.png")
    plot_curves(history, FIG_DIR / "gan_training_curves.png")

    METRICS_PATH.write_text(json.dumps({
        "model": "GAN (Assignment 3 architecture)", "dataset": "MNIST",
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
        "optimizer": "Adam (betas 0.5, 0.999)", "loss": "BCE (non-saturating generator loss)",
        "generator_parameters": sum(p.numel() for p in G.parameters()),
        "discriminator_parameters": sum(p.numel() for p in D.parameters()),
        "final_d_loss": round(history["d_loss"][-1], 4), "final_g_loss": round(history["g_loss"][-1], 4),
        "final_d_real": round(history["d_real"][-1], 4), "final_d_fake": round(history["d_fake"][-1], 4),
        "history": {k: [round(v, 4) for v in vals] for k, vals in history.items()},
    }, indent=2))
    print(f"Saved {GEN_PATH}, {DISC_PATH}, {METRICS_PATH} and sample grids in {FIG_DIR}/")


if __name__ == "__main__":
    main()
