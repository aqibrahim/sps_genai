"""Train the Assignment 4 energy-based model on CIFAR-10 (contrastive divergence + Langevin sampling).

Usage:  uv run --with matplotlib python train_ebm_cifar10.py            (default 10 epochs)
        uv run --with matplotlib python train_ebm_cifar10.py --epochs 20
"""
import argparse
import json
import time
from pathlib import Path

import torch

from helper_lib.data_loader import get_cifar10_generation_loader
from helper_lib.generator import generate_ebm_samples, save_samples
from helper_lib.model import EnergyModel
from helper_lib.trainer import train_ebm
from helper_lib.utils import get_device

MODEL_PATH = Path("models/ebm.pth")
METRICS_PATH = Path("models/ebm_metrics.json")
FIG_DIR = Path("docs/screenshots")
SNAPSHOT_EPOCHS = {1, 3, 5}  # plus the final epoch
LANGEVIN = dict(steps=60, step_size=10.0, noise_std=0.005)


def plot_curves(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["loss"], label="Total loss")
    ax1.plot(epochs, history["cd_loss"], label="Contrastive divergence")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("EBM losses"); ax1.legend()
    ax2.plot(epochs, history["energy_real"], label="E(real images)")
    ax2.plot(epochs, history["energy_fake"], label="E(Langevin samples)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Mean energy"); ax2.set_title("Energies"); ax2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    loader = get_cifar10_generation_loader(args.data_dir, batch_size=args.batch_size)
    model = EnergyModel(in_channels=3)
    print(f"Energy model parameters: {sum(p.numel() for p in model.parameters()):,}")
    start = time.time()

    def snapshot(epoch, model, buffer):
        print(f"  elapsed: {(time.time() - start) / 60:.1f} min")
        if epoch in SNAPSHOT_EPOCHS or epoch % 10 == 0 or epoch == args.epochs:
            images, _, _ = generate_ebm_samples(model, num_samples=64, steps=200, device=device, seed=args.seed)
            save_samples(images, FIG_DIR / f"ebm_epoch_{epoch:02d}.png")
            # Replay-buffer samples: chains that have been refined over many training iterations
            # (the standard way EBM papers show samples)
            idx = torch.randperm(buffer.size(0), device=buffer.device)[:64]
            save_samples(((buffer[idx] + 1) / 2).clamp(0, 1).cpu(), FIG_DIR / f"ebm_buffer_epoch_{epoch:02d}.png")

    history = train_ebm(model, loader, device=device, epochs=args.epochs, lr=args.lr,
                        langevin_steps=LANGEVIN["steps"], step_size=LANGEVIN["step_size"],
                        noise_std=LANGEVIN["noise_std"], on_epoch_end=snapshot)
    model = model.to("cpu")
    torch.save({"state_dict": model.state_dict(), "epochs": args.epochs, "langevin": LANGEVIN}, MODEL_PATH)

    images, e_before, e_after = generate_ebm_samples(model, num_samples=64, steps=200, seed=7)
    save_samples(images, FIG_DIR / "ebm_samples_final.png")
    plot_curves(history, FIG_DIR / "ebm_training_curves.png")

    METRICS_PATH.write_text(json.dumps({
        "model": "Energy-based model (CNN energy function, contrastive divergence, Langevin sampling)",
        "dataset": "CIFAR-10", "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
        "optimizer": "Adam (betas 0.0, 0.999)", "regularization_alpha": 0.1, "langevin_training": LANGEVIN,
        "replay_buffer": {"size": 8192, "reinit_probability": 0.05},
        "parameters": sum(p.numel() for p in model.parameters()),
        "generation_from_noise": {"langevin_steps": 200,
                                  "mean_energy_before": round(sum(e_before) / len(e_before), 4),
                                  "mean_energy_after": round(sum(e_after) / len(e_after), 4)},
        "training_minutes": round((time.time() - start) / 60, 1),
        "history": {k: [round(v, 4) for v in vals] for k, vals in history.items()},
    }, indent=2))
    print(f"Saved {MODEL_PATH}, {METRICS_PATH} and sample grids in {FIG_DIR}/")


if __name__ == "__main__":
    main()
