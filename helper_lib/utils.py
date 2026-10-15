"""Saving, loading and device helpers."""
import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return "mps"
    return "cpu"


def save_model(model, path, **metadata):
    """Save weights plus metadata (class names, accuracy, ...) in one file."""
    torch.save({"state_dict": model.state_dict(), **metadata}, path)


def load_model(model, path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
