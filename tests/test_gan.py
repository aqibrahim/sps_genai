"""Tests for the Assignment 3 GAN architecture and the /gan endpoints."""
import io

import pytest
import torch
from PIL import Image

from helper_lib.generator import generate_gan_samples
from helper_lib.model import Discriminator, Generator, get_model


# ---------- Architecture ----------
def test_generator_layer_shapes():
    G = Generator()
    z = torch.randn(2, 100)
    x = G.fc(z)
    assert x.shape == (2, 7 * 7 * 128)
    x = x.view(-1, 128, 7, 7)
    expected = [
        (2, 64, 14, 14),  # ConvTranspose2d 128->64, k4, s2, p1
        (2, 64, 14, 14),  # BatchNorm2d
        (2, 64, 14, 14),  # ReLU
        (2, 1, 28, 28),   # ConvTranspose2d 64->1, k4, s2, p1
        (2, 1, 28, 28),   # Tanh
    ]
    for layer, shape in zip(G.net, expected):
        x = layer(x)
        assert tuple(x.shape) == shape, f"{layer} gave {tuple(x.shape)}"


def test_generator_layer_types_and_settings():
    G = Generator()
    convt1, bn, relu, convt2, tanh = list(G.net)
    assert (convt1.in_channels, convt1.out_channels) == (128, 64)
    assert (convt2.in_channels, convt2.out_channels) == (64, 1)
    for c in (convt1, convt2):
        assert c.kernel_size == (4, 4) and c.stride == (2, 2) and c.padding == (1, 1)
    assert isinstance(bn, torch.nn.BatchNorm2d) and isinstance(relu, torch.nn.ReLU)
    assert isinstance(tanh, torch.nn.Tanh)


def test_generator_output_range():
    out = Generator()(torch.randn(8, 100))
    assert out.shape == (8, 1, 28, 28)
    assert out.min() >= -1 and out.max() <= 1


def test_discriminator_layer_shapes():
    D = Discriminator()
    x = torch.randn(2, 1, 28, 28)
    expected = [
        (2, 64, 14, 14),   # Conv2d 1->64, k4, s2, p1
        (2, 64, 14, 14),   # LeakyReLU(0.2)
        (2, 128, 7, 7),    # Conv2d 64->128, k4, s2, p1
        (2, 128, 7, 7),    # BatchNorm2d
        (2, 128, 7, 7),    # LeakyReLU(0.2)
        (2, 6272),         # Flatten
        (2, 1),            # Linear -> single output
        (2, 1),            # Sigmoid -> probability
    ]
    for layer, shape in zip(D.net, expected):
        x = layer(x)
        assert tuple(x.shape) == shape, f"{layer} gave {tuple(x.shape)}"


def test_discriminator_settings_and_probability():
    D = Discriminator()
    conv1, lrelu1, conv2, bn, lrelu2, _, linear, _ = list(D.net)
    assert (conv1.in_channels, conv1.out_channels) == (1, 64)
    assert (conv2.in_channels, conv2.out_channels) == (64, 128)
    for c in (conv1, conv2):
        assert c.kernel_size == (4, 4) and c.stride == (2, 2) and c.padding == (1, 1)
    assert lrelu1.negative_slope == 0.2 and lrelu2.negative_slope == 0.2
    assert isinstance(bn, torch.nn.BatchNorm2d)
    assert (linear.in_features, linear.out_features) == (6272, 1)
    p = D(torch.randn(4, 1, 28, 28))
    assert p.shape == (4, 1) and ((p > 0) & (p < 1)).all()


def test_parameter_counts():
    # G: fc 633,472 + convT1 131,136 + BN 128 + convT2 1,025
    assert sum(p.numel() for p in Generator().parameters()) == 765_761
    # D: conv1 1,088 + conv2 131,200 + BN 256 + linear 6,273
    assert sum(p.numel() for p in Discriminator().parameters()) == 138_817


def test_get_model_gan():
    gan = get_model("GAN")
    assert isinstance(gan.generator, Generator) and isinstance(gan.discriminator, Discriminator)


def test_generate_samples_seed_reproducible():
    G = Generator()
    a = generate_gan_samples(G, num_samples=4, seed=123)
    b = generate_gan_samples(G, num_samples=4, seed=123)
    assert torch.equal(a, b)
    assert a.shape == (4, 1, 28, 28) and a.min() >= 0 and a.max() <= 1


# ---------- API ----------
def test_gan_generate_png(client):
    r = client.get("/gan/generate?num_images=16&seed=1&scale=2")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    image = Image.open(io.BytesIO(r.content))
    # 4x4 grid of 28px digits with 2px padding: 4*28 + 5*2 = 122 px, times scale 2
    assert image.size == (244, 244)


def test_gan_generate_same_seed_same_image(client):
    a = client.get("/gan/generate?num_images=4&seed=5").content
    b = client.get("/gan/generate?num_images=4&seed=5").content
    assert a == b


def test_gan_generate_json(client):
    r = client.post("/gan/generate", json={"num_images": 9, "seed": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["num_images"] == 9 and body["image_shape"] == [1, 28, 28]
    assert len(body["discriminator_scores"]) == 9
    assert all(0 <= s <= 1 for s in body["discriminator_scores"])
    import base64
    Image.open(io.BytesIO(base64.b64decode(body["image_base64"])))  # valid PNG


@pytest.mark.parametrize("query", ["num_images=0", "num_images=65", "scale=9"])
def test_gan_generate_validation(client, query):
    assert client.get(f"/gan/generate?{query}").status_code == 422


def test_gan_info(client):
    body = client.get("/gan/info").json()
    assert body["latent_dim"] == 100 and body["output_shape"] == [1, 28, 28]
