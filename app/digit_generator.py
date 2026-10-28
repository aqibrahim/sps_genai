"""MNIST digit generator used by the API (Assignment 3)."""
import base64
import io
from pathlib import Path

import torch
from PIL import Image

from helper_lib.generator import generate_gan_samples, make_image_grid
from helper_lib.model import Discriminator, Generator

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
GENERATOR_PATH = MODELS_DIR / "gan_generator.pth"
DISCRIMINATOR_PATH = MODELS_DIR / "gan_discriminator.pth"


class DigitGenerator:
    def __init__(self, generator_path=GENERATOR_PATH, discriminator_path=DISCRIMINATOR_PATH):
        ckpt = torch.load(generator_path, map_location="cpu", weights_only=True)
        self.generator = Generator(ckpt.get("latent_dim", 100))
        self.generator.load_state_dict(ckpt["state_dict"])
        self.generator.eval()
        self.epochs = ckpt.get("epochs")

        self.discriminator = None  # optional: used to score how "real" each sample looks
        if Path(discriminator_path).exists():
            d_ckpt = torch.load(discriminator_path, map_location="cpu", weights_only=True)
            self.discriminator = Discriminator()
            self.discriminator.load_state_dict(d_ckpt["state_dict"])
            self.discriminator.eval()

    def generate(self, num_images=16, seed=None):
        """Images in [0, 1], shape (N, 1, 28, 28)."""
        return generate_gan_samples(self.generator, num_samples=num_images, seed=seed)

    def realness_scores(self, images):
        """Discriminator probability that each image is a real MNIST digit."""
        if self.discriminator is None:
            return None
        with torch.no_grad():
            scores = self.discriminator(images * 2 - 1)  # back to the [-1, 1] training range
        return [round(s, 4) for s in scores.squeeze(1).tolist()]

    @staticmethod
    def to_png(images, scale=4):
        """Grid of images as PNG bytes, upscaled so the 28x28 digits are easy to see."""
        grid = make_image_grid(images)  # (C, H, W) in [0, 1]
        array = (grid[0] * 255).round().clamp(0, 255).to(torch.uint8).numpy()
        image = Image.fromarray(array)  # 2-D uint8 array -> grayscale
        if scale > 1:
            image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def to_base64(png_bytes):
        return base64.b64encode(png_bytes).decode("ascii")
