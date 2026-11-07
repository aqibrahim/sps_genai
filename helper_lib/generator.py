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


def images_to_png(images, scale=4):
    """Grid of images (N, 1, H, W) in [0, 1] as PNG bytes, upscaled so small digits are easy to see."""
    import io

    from PIL import Image

    grid = make_image_grid(images)  # (C, H, W) in [0, 1]
    array = (grid[0] * 255).round().clamp(0, 255).to(torch.uint8).numpy()
    image = Image.fromarray(array)  # 2-D uint8 array -> grayscale
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def discriminator_scores(discriminator, images):
    """Discriminator probability that each image (in [0, 1]) is a real training image."""
    discriminator.eval()
    with torch.no_grad():
        scores = discriminator(images * 2 - 1)  # back to the [-1, 1] training range
    return [round(s, 4) for s in scores.squeeze(1).tolist()]
