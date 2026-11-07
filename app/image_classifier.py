"""CIFAR-10 inference: preprocessing and prediction for a trained CNN (Assignment 2).

The CNN itself (helper_lib.model.CNN) is built and loaded in app/main.py and passed in here.
"""
import io

import torch
from PIL import Image
from torchvision import transforms

from helper_lib.data_loader import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD


class ImageClassifier:
    def __init__(self, model, classes=None, test_accuracy=None, device="cpu"):
        self.model = model.to(device).eval()
        self.device = device
        self.classes = classes or CIFAR10_CLASSES
        self.test_accuracy = test_accuracy
        # Training images were 32x32 CIFAR images upscaled to 64x64. Uploaded photos are
        # first shrunk to 32x32 and then upscaled the same way, so they look like training data.
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def predict(self, image_bytes: bytes, top_k: int = 3) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)  # shape 1x3x64x64
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
        top = torch.topk(probs, k=min(top_k, len(self.classes)))
        return {
            "predicted_class": self.classes[top.indices[0].item()],
            "class_index": top.indices[0].item(),
            "confidence": round(top.values[0].item(), 4),
            "top_predictions": [
                {"class": self.classes[i.item()], "probability": round(p.item(), 4)}
                for p, i in zip(top.values, top.indices)
            ],
        }
