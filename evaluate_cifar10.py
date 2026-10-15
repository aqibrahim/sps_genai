"""Detailed evaluation of the trained CIFAR-10 CNN (no retraining).

Computes overall and per-class test accuracy, a confusion matrix, and the most
frequently confused class pairs, and saves charts for the README.

Usage:  uv run --with matplotlib python evaluate_cifar10.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt

from helper_lib.data_loader import CIFAR10_CLASSES, get_data_loader
from helper_lib.evaluator import confusion_matrix, get_predictions, per_class_accuracy
from helper_lib.model import get_model
from helper_lib.utils import get_device, load_model

MODEL_PATH = Path("models/cifar10_cnn.pth")
RESULTS_PATH = Path("models/cifar10_cnn_evaluation.json")
FIG_DIR = Path("docs/screenshots")


def plot_confusion_matrix(matrix, path):
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(10), CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CIFAR10_CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("CIFAR-10 confusion matrix (10,000 test images)")
    threshold = matrix.max() / 2
    for i in range(10):
        for j in range(10):
            value = int(matrix[i, j])
            ax.text(j, i, value, ha="center", va="center", fontsize=9,
                    color="white" if value > threshold else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_accuracy(accuracies, overall, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(CIFAR10_CLASSES, accuracies, color="#4C72B0")
    ax.axhline(overall, color="#C44E52", linestyle="--", label=f"Overall: {overall:.2f}%")
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 1, f"{acc:.1f}", ha="center", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Per-class test accuracy")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    device = get_device()
    print(f"Using device: {device}")
    model, _ = load_model(get_model("CNN"), MODEL_PATH, device=device)
    test_loader = get_data_loader("data", batch_size=256, train=False)

    preds, labels = get_predictions(model, test_loader, device=device)
    matrix = confusion_matrix(preds, labels)
    accuracies = per_class_accuracy(matrix)
    overall = 100 * matrix.diag().sum().item() / matrix.sum().item()

    print(f"\nOverall test accuracy: {overall:.2f}%\n")
    print(f"{'Class':<12}{'Accuracy':>10}")
    for name, acc in sorted(zip(CIFAR10_CLASSES, accuracies), key=lambda x: -x[1]):
        print(f"{name:<12}{acc:>9.1f}%")

    confusions = [
        (int(matrix[i, j]), CIFAR10_CLASSES[i], CIFAR10_CLASSES[j])
        for i in range(10) for j in range(10) if i != j
    ]
    confusions.sort(reverse=True)
    print("\nMost common mistakes (true -> predicted):")
    for count, true_name, pred_name in confusions[:5]:
        print(f"  {true_name} -> {pred_name}: {count} images")

    RESULTS_PATH.write_text(json.dumps({
        "overall_accuracy": round(overall, 2),
        "per_class_accuracy": {n: round(a, 1) for n, a in zip(CIFAR10_CLASSES, accuracies)},
        "top_confusions": [{"true": t, "predicted": p, "count": c} for c, t, p in confusions[:5]],
        "confusion_matrix": matrix.tolist(),
        "class_order": CIFAR10_CLASSES,
    }, indent=2))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(matrix, FIG_DIR / "confusion_matrix.png")
    plot_per_class_accuracy(accuracies, overall, FIG_DIR / "per_class_accuracy.png")
    print(f"\nSaved {RESULTS_PATH}, {FIG_DIR}/confusion_matrix.png and {FIG_DIR}/per_class_accuracy.png")


if __name__ == "__main__":
    main()
