"""Train the Assignment 2 CNN on CIFAR-10 and save it for the API.

Usage:  uv run python train_cifar10.py            (default 10 epochs)
        uv run python train_cifar10.py --epochs 15
"""
import argparse
import json
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from helper_lib.data_loader import CIFAR10_CLASSES, get_data_loader
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.utils import get_device, save_model

MODEL_PATH = Path("models/cifar10_cnn.pth")
METRICS_PATH = Path("models/cifar10_cnn_metrics.json")
SAMPLES_DIR = Path("sample_images")


def export_sample_images(data_dir):
    """Save one CIFAR-10 test image per class, to try the API with."""
    SAMPLES_DIR.mkdir(exist_ok=True)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True)
    saved = set()
    for image, label in test_set:  # PIL images, 32x32
        name = CIFAR10_CLASSES[label]
        if name not in saved:
            image.resize((128, 128)).save(SAMPLES_DIR / f"{name}.png")
            saved.add(name)
        if len(saved) == len(CIFAR10_CLASSES):
            break
    print(f"Saved {len(saved)} sample images to {SAMPLES_DIR}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_loader = get_data_loader(args.data_dir, batch_size=args.batch_size, train=True)
    test_loader = get_data_loader(args.data_dir, batch_size=args.batch_size, train=False)

    model = get_model("CNN")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model = train_model(model, train_loader, criterion, optimizer, device=device, epochs=args.epochs)
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device=device)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.to("cpu")
    save_model(model, MODEL_PATH, classes=CIFAR10_CLASSES, image_size=64,
               test_accuracy=round(test_acc, 2), epochs=args.epochs)
    METRICS_PATH.write_text(json.dumps({
        "model": "CNN (Assignment 2 architecture)", "dataset": "CIFAR-10",
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
        "test_loss": round(test_loss, 4), "test_accuracy": round(test_acc, 2),
    }, indent=2))
    print(f"Saved model to {MODEL_PATH} and metrics to {METRICS_PATH}")

    export_sample_images(args.data_dir)


if __name__ == "__main__":
    main()
