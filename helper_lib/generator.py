"""Image generation helpers (Module 5/6 activities)."""
import math

import torch
from torchvision.utils import make_grid, save_image


def generate_gan_samples(generator, num_samples=16, device="cpu", seed=None):
    """Sample noise, run the generator, and return images scaled to [0, 1], shape (N, 1, 28, 28)."""
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    z = torch.randn(num_samples, generator.latent_dim, generator=gen).to(device)
    generator.eval()  # batch norm uses its running statistics at inference
    with torch.no_grad():
        images = generator(z.to(device)).cpu()
    return (images + 1) / 2  # Tanh range [-1, 1] -> [0, 1]


def make_image_grid(images, padding=2):
    """Arrange N images in a near-square grid, returned as a (C, H, W) tensor in [0, 1]."""
    nrow = math.ceil(math.sqrt(images.size(0)))
    return make_grid(images, nrow=nrow, padding=padding, pad_value=1.0)


def save_samples(images, path):
    save_image(make_image_grid(images), path)
