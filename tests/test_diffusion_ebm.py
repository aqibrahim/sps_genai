"""Tests for the Assignment 4 diffusion model and energy-based model."""
import io
import math
from pathlib import Path

import pytest
import torch
from PIL import Image

from helper_lib.generator import generate_diffusion_samples, generate_ebm_samples, langevin_sample
from helper_lib.model import DiffusionModel, EnergyModel, SinusoidalEmbedding, UNet, get_model
from helper_lib.trainer import train_diffusion, train_ebm

MODELS = Path(__file__).resolve().parent.parent / "models"


def tiny_loader(n=8, batch=4):
    data = torch.rand(n, 3, 32, 32) * 2 - 1
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, torch.zeros(n)), batch_size=batch)


# ---------- Diffusion ----------
def test_sinusoidal_embedding_matches_question_2():
    """t = 1, dimension 8, max period 10000 -> frequencies 1, 0.1, 0.01, 0.001."""
    emb = SinusoidalEmbedding(dim=8, max_period=10000)(torch.tensor([1]))[0]
    expected = [math.sin(1), math.sin(0.1), math.sin(0.01), math.sin(0.001),
                math.cos(1), math.cos(0.1), math.cos(0.01), math.cos(0.001)]
    assert torch.allclose(emb, torch.tensor(expected), atol=1e-6)


def test_unet_predicts_noise_with_input_shape():
    unet = UNet()
    x = torch.randn(2, 3, 32, 32)
    assert unet(x, torch.tensor([0, 999])).shape == (2, 3, 32, 32)


def test_unet_resolutions():
    """Down path 32 -> 16 -> 8 -> 4 (bottleneck), then back up to 32."""
    unet = UNet()
    shapes = []
    for block in list(unet.down) + [unet.bottleneck] + list(unet.up):
        block.register_forward_hook(lambda m, i, o: shapes.append(tuple(o.shape[-3:])))
    unet(torch.randn(1, 3, 32, 32), torch.tensor([10]))
    assert shapes == [(32, 16, 16), (64, 8, 8), (96, 4, 4), (128, 4, 4), (96, 8, 8), (64, 16, 16), (32, 32, 32)]


def test_noise_schedule_and_forward_process():
    d = DiffusionModel(UNet(), timesteps=1000)
    a = d.alpha_bar
    assert a[0] > 0.999 and a[-1] < 1e-3 and (a[1:] < a[:-1]).all()  # signal fades to ~0
    x0, noise = torch.rand(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
    assert torch.allclose(d.add_noise(x0, torch.tensor([0, 0]), noise), x0, atol=0.05)
    assert d.training_loss(x0).ndim == 0


def test_train_diffusion_smoke():
    """Class-activity signature: train_diffusion(model, data_loader, criterion, optimizer, device, epochs)."""
    model = get_model("Diffusion")
    optimizer = torch.optim.AdamW(model.unet.parameters(), lr=1e-3)
    out = train_diffusion(model, tiny_loader(), torch.nn.MSELoss(), optimizer, device="cpu", epochs=1)
    assert out is model and len(model.history["loss"]) == 1 and isinstance(model.ema_unet, UNet)


def test_generate_samples_activity_interface():
    """Class-activity signature: generate_samples(model, device, num_samples, diffusion_steps)."""
    from helper_lib.generator import generate_samples
    from helper_lib.model import VAE

    images = generate_samples(get_model("Diffusion"), "cpu", num_samples=4, diffusion_steps=5, show=False)
    assert images.shape == (4, 3, 32, 32)
    assert generate_samples(get_model("GAN"), "cpu", num_samples=2, show=False).shape == (2, 1, 28, 28)
    assert generate_samples(VAE(), "cpu", num_samples=2, show=False).shape == (2, 1, 28, 28)


def test_vae_shapes():
    from helper_lib.model import VAE
    recon, mu, logvar = get_model("VAE")(torch.rand(3, 1, 28, 28))
    assert recon.shape == (3, 1, 28, 28) and mu.shape == logvar.shape == (3, 16)


def test_ddim_sampling_shape_and_range():
    images = generate_diffusion_samples(DiffusionModel(UNet()), num_samples=3, steps=5, seed=0)
    assert images.shape == (3, 3, 32, 32) and images.min() >= 0 and images.max() <= 1


# ---------- Energy-based model ----------
def test_energy_model_shape_and_parameters():
    model = EnergyModel()
    assert model(torch.randn(5, 3, 32, 32)).shape == (5,)
    # convs 1,216 + 4,640 + 18,496 + 36,928; linear 16,448 + 65
    assert sum(p.numel() for p in model.parameters()) == 77_793


def test_langevin_updates_inputs_not_weights():
    """Langevin dynamics computes gradients for the images only; weights get no gradient."""
    torch.manual_seed(0)
    model = EnergyModel()
    x = torch.rand(8, 3, 32, 32) * 2 - 1
    with torch.no_grad():
        before = model(x).sum().item()
    out = langevin_sample(model, x, steps=5, step_size=0.5, noise_std=0.0)
    with torch.no_grad():
        after = model(out).sum().item()
    assert after < before                                      # energy went down
    assert not torch.equal(out, x)                             # the images changed
    assert all(p.grad is None for p in model.parameters())     # no weight gradients
    assert all(p.requires_grad for p in model.parameters())    # weights trainable again afterwards
    assert not out.requires_grad and out.min() >= -1 and out.max() <= 1


def test_train_ebm_smoke():
    history = train_ebm(EnergyModel(), tiny_loader(), epochs=1, langevin_steps=2, buffer_size=16)
    assert len(history["loss"]) == 1


def test_generate_ebm_samples():
    images, before, after = generate_ebm_samples(EnergyModel(), num_samples=4, steps=3, seed=1)
    assert images.shape == (4, 3, 32, 32) and len(before) == len(after) == 4


def test_get_model_new_names():
    from helper_lib.model import VAE
    assert isinstance(get_model("VAE"), VAE)
    assert isinstance(get_model("EBM"), EnergyModel)
    assert isinstance(get_model("Diffusion"), DiffusionModel)


# ---------- API (run once the models are trained) ----------
needs_diffusion = pytest.mark.skipif(not (MODELS / "diffusion_unet.pth").exists(), reason="diffusion model not trained yet")
needs_ebm = pytest.mark.skipif(not (MODELS / "ebm.pth").exists(), reason="energy model not trained yet")


@needs_diffusion
def test_api_diffusion(client):
    from app import main
    assert isinstance(main.diffusion, DiffusionModel) and isinstance(main.diffusion.unet, UNet)
    r = client.get("/diffusion/generate?num_images=4&steps=10&seed=1&scale=1")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(r.content)).size == (2 * 32 + 3 * 2, 2 * 32 + 3 * 2)
    assert r.content == client.get("/diffusion/generate?num_images=4&steps=10&seed=1&scale=1").content
    body = client.post("/diffusion/generate", json={"num_images": 2, "steps": 10, "seed": 3}).json()
    assert body["image_shape"] == [3, 32, 32] and body["image_base64"]
    assert client.get("/diffusion/generate?steps=2").status_code == 422
    assert client.get("/diffusion/info").json()["unet"]["resolutions"] == [32, 16, 8, 4]


@needs_ebm
def test_api_ebm(client):
    from app import main
    assert isinstance(main.energy_model, EnergyModel)
    r = client.get("/ebm/generate?num_images=4&steps=20&seed=1")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    body = client.post("/ebm/generate", json={"num_images": 4, "steps": 50, "seed": 2}).json()
    assert len(body["energy_after"]) == 4
    assert body["mean_energy_after"] < body["mean_energy_before"]  # Langevin lowers the energy
    assert client.get("/ebm/generate?num_images=0").status_code == 422
    assert client.get("/ebm/info").json()["energy_model"]["parameters"] == 77_793
