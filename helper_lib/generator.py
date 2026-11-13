"""Image generation helpers: GAN (Module 6), diffusion and energy-based sampling (Module 8)."""
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
    """Grid of images (N, C, H, W) in [0, 1] (grayscale or RGB) as PNG bytes, upscaled so small images are easy to see."""
    import io

    from PIL import Image

    grid = make_image_grid(images)  # (3, H, W) in [0, 1]; make_grid repeats grayscale to 3 channels
    array = (grid * 255).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    image = Image.fromarray(array)  # (H, W, 3) uint8 -> RGB
    if images.size(1) == 1:
        image = image.convert("L")
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


# ---------- Diffusion: DDIM sampling (Module 8) ----------
def generate_diffusion_samples(diffusion, num_samples=16, steps=50, device="cpu", seed=None, image_size=32):
    """Reverse diffusion with the deterministic DDIM sampler (Song et al., 2021).

    Start from pure noise x_T ~ N(0, I) and walk back over `steps` evenly spaced timesteps.
    At each timestep t (next one s < t):
        eps   = UNet(x_t, t)                                              (predicted noise)
        x0    = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t)                   (predicted clean image)
        x_s   = sqrt(a_s) * x0 + sqrt(1 - a_s) * eps                      (move to the less noisy step)
    where a_t = alpha_bar_t. Fewer steps than the 1,000 training steps give fast generation.
    Returns images in [0, 1], shape (N, 3, 32, 32).
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    x = torch.randn(num_samples, 3, image_size, image_size, generator=gen).to(device)
    diffusion.eval()
    T = diffusion.timesteps
    times = torch.linspace(T - 1, 0, steps).round().long().tolist()
    with torch.no_grad():
        for i, t in enumerate(times):
            a_t = diffusion.alpha_bar[t]
            a_s = diffusion.alpha_bar[times[i + 1]] if i + 1 < len(times) else torch.tensor(1.0, device=device)
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            eps = diffusion.unet(x, t_batch)
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
            x = a_s.sqrt() * x0 + (1 - a_s).sqrt() * eps
    return ((x.clamp(-1, 1) + 1) / 2).cpu()


# ---------- Energy-based model: Langevin dynamics (Module 8) ----------
def langevin_sample(model, x, steps=60, step_size=10.0, noise_std=0.005, grad_clip=0.03):
    """Langevin dynamics: move images *downhill in energy* by gradient descent on the input pixels.

        x <- x + noise                      (noise ~ N(0, noise_std^2): explores, avoids collapsing to one mode)
        x <- x - step_size * dE(x)/dx       (drift towards lower energy)

    Only the images require gradients: the model's weights are frozen during sampling, so
    .backward() computes dE/dx and leaves the weight gradients untouched. Each step starts
    from a detached copy so the computational graph does not grow across steps.
    """
    was_training = model.training
    model.eval()
    requires_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)                       # freeze weights: no gradients for them

    x = x.detach().clone()
    with torch.enable_grad():
        for _ in range(steps):
            x = (x + noise_std * torch.randn_like(x)).clamp(-1, 1)
            x.requires_grad_(True)                    # track gradients for the input pixels
            energy = model(x).sum()
            grad, = torch.autograd.grad(energy, x)     # dE/dx only
            x = (x - step_size * grad.clamp(-grad_clip, grad_clip)).clamp(-1, 1).detach()

    for p, flag in zip(model.parameters(), requires_grad):
        p.requires_grad_(flag)                        # restore for training
    model.train(was_training)
    return x


def generate_ebm_samples(model, num_samples=16, steps=200, step_size=10.0, noise_std=0.005,
                         device="cpu", seed=None, image_size=32):
    """Start from uniform noise and run Langevin dynamics. Returns (images in [0, 1], energies before, after)."""
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    x = (torch.rand(num_samples, 3, image_size, image_size, generator=gen) * 2 - 1).to(device)
    if seed is not None:
        torch.manual_seed(seed)  # makes the Langevin noise reproducible too
    model.eval()
    with torch.no_grad():
        energy_before = model(x).tolist()
    x = langevin_sample(model, x, steps=steps, step_size=step_size, noise_std=noise_std)
    with torch.no_grad():
        energy_after = model(x).tolist()
    return ((x + 1) / 2).cpu(), energy_before, energy_after
