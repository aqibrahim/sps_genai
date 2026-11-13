"""Model definitions. get_model() returns one of: FCNN, CNN, EnhancedCNN, VAE, GAN, Diffusion, EBM."""
import math

import torch
import torch.nn as nn


class FCNN(nn.Module):
    """Fully connected baseline (Module 4)."""

    def __init__(self, input_size=64 * 64 * 3, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """Assignment 2 architecture.

    Input 64x64x3
    -> Conv2d(3->16, 3x3, stride 1, pad 1) -> ReLU -> MaxPool(2x2, stride 2)   : 16 x 32 x 32
    -> Conv2d(16->32, 3x3, stride 1, pad 1) -> ReLU -> MaxPool(2x2, stride 2)  : 32 x 16 x 16
    -> Flatten (32*16*16 = 8192) -> Linear(8192->100) -> ReLU -> Linear(100->10)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class EnhancedCNN(nn.Module):
    """Deeper CNN with batch norm and dropout (Module 4)."""

    def __init__(self, num_classes=10):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))  # 64 -> 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------- Module 5: Variational autoencoder (28x28 grayscale) ----------
class VAE(nn.Module):
    """Convolutional VAE (Module 5 activity).

    Encoder: (B, 1, 28, 28) -> Conv(1->32, k3, s2, p1) + ReLU (14x14) -> Conv(32->64, k3, s2, p1) + ReLU (7x7)
             -> Flatten (3136) -> Linear to mu and log-variance (latent_dim each)
    Reparameterization: z = mu + exp(0.5 * logvar) * eps,  eps ~ N(0, I)
    Decoder: Linear(latent_dim -> 3136) -> reshape (64, 7, 7) -> ConvT(64->32, k4, s2, p1) + ReLU (14x14)
             -> ConvT(32->1, k4, s2, p1) + Sigmoid (28x28)
    """

    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(self.fc_decode(z).view(-1, 64, 7, 7))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z), mu, logvar


# ---------- Assignment 3: GAN for 28x28 MNIST digits ----------
LATENT_DIM = 100


class Generator(nn.Module):
    """Assignment 3 generator.

    Noise (B, 100)
    -> Linear(100 -> 7*7*128) -> reshape to (B, 128, 7, 7)
    -> ConvTranspose2d(128 -> 64, k4, s2, p1) -> BatchNorm2d -> ReLU    : (B, 64, 14, 14)
    -> ConvTranspose2d(64 -> 1,  k4, s2, p1) -> Tanh                    : (B, 1, 28, 28)
    Tanh keeps pixels in [-1, 1], matching the normalized MNIST images.
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 7, 7)
        return self.net(x)


class Discriminator(nn.Module):
    """Assignment 3 discriminator.

    Image (B, 1, 28, 28)
    -> Conv2d(1 -> 64, k4, s2, p1) -> LeakyReLU(0.2)                    : (B, 64, 14, 14)
    -> Conv2d(64 -> 128, k4, s2, p1) -> BatchNorm2d -> LeakyReLU(0.2)   : (B, 128, 7, 7)
    -> Flatten (6272) -> Linear(6272 -> 1) -> Sigmoid                   : (B, 1)
    The sigmoid turns the single linear output into the real/fake probability.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class GAN(nn.Module):
    """Container holding the generator and discriminator together."""

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()


def init_gan_weights(module):
    """DCGAN-style initialization: N(0, 0.02) for conv/linear, N(1, 0.02) for batch norm."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


# ---------- Assignment 4: Energy-based model for CIFAR-10 (3x32x32) ----------
class EnergyModel(nn.Module):
    """Maps an image to a single scalar energy E(x): low for realistic images, high otherwise.

    Image (B, 3, 32, 32)
    -> Conv2d(3 -> 16, k5, s2, p2) + Swish     : (B, 16, 16, 16)
    -> Conv2d(16 -> 32, k3, s2, p1) + Swish    : (B, 32, 8, 8)
    -> Conv2d(32 -> 64, k3, s2, p1) + Swish    : (B, 64, 4, 4)
    -> Conv2d(64 -> 64, k3, s2, p1) + Swish    : (B, 64, 2, 2)
    -> Flatten (256) -> Linear(256 -> 64) + Swish -> Linear(64 -> 1)   : energy (B,)
    Swish (SiLU) is smooth, so the gradient of the energy with respect to the input
    (used by Langevin sampling) is smooth as well.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------- Assignment 4: Diffusion model (DDPM noise prediction + U-Net) ----------
class SinusoidalEmbedding(nn.Module):
    """Sinusoidal timestep embedding (same formula as transformer positional encoding).

    For embedding dimension d, frequencies w_i = max_period^(-i / (d/2)), i = 0 .. d/2 - 1:
        emb(t) = [sin(t*w_0), ..., sin(t*w_{d/2-1}), cos(t*w_0), ..., cos(t*w_{d/2-1})]
    """

    def __init__(self, dim=32, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, dim)


class ResidualBlock(nn.Module):
    """BatchNorm -> Conv3x3 + Swish -> Conv3x3, plus a (1x1-projected) skip connection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.norm = nn.BatchNorm2d(in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.conv1(self.norm(x)))
        return self.conv2(h) + self.skip(x)


class DownBlock(nn.Module):
    """block_depth residual blocks (each output saved for a skip connection), then AvgPool2d(2): halves H, W."""

    def __init__(self, in_channels, out_channels, block_depth=2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResidualBlock(in_channels if i == 0 else out_channels, out_channels) for i in range(block_depth)]
        )
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return self.pool(x)


class UpBlock(nn.Module):
    """Upsample(2) (doubles H, W), then residual blocks that each consume one saved skip connection."""

    def __init__(self, in_channels, skip_channels, out_channels, block_depth=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.blocks = nn.ModuleList(
            [ResidualBlock((in_channels if i == 0 else out_channels) + skip_channels, out_channels)
             for i in range(block_depth)]
        )

    def forward(self, x, skips):
        x = self.up(x)
        for block in self.blocks:
            x = block(torch.cat([x, skips.pop()], dim=1))
        return x


class UNet(nn.Module):
    """U-Net denoiser: given a noisy image x_t and its timestep t, predicts the noise eps that was added.

    1. t -> sinusoidal embedding (32 dims) -> upsampled to 32x32 and concatenated with the image features
    2. Down path:   32x32 (32 ch) -> 16x16 (64 ch) -> 8x8 (96 ch) -> 4x4   (AvgPool2d after each DownBlock)
    3. Bottleneck:  two residual blocks at 4x4 with 128 channels
    4. Up path:     4x4 -> 8x8 -> 16x16 -> 32x32, each UpBlock concatenating the matching skip connections
    5. Conv1x1 -> 3 channels: predicted noise, same shape as the input image
    """

    def __init__(self, image_size=32, in_channels=3, widths=(32, 64, 96, 128), block_depth=2, embedding_dim=32):
        super().__init__()
        self.image_size = image_size
        self.time_embedding = SinusoidalEmbedding(embedding_dim)
        self.embedding_upsample = nn.Upsample(size=image_size, mode="nearest")
        self.input_conv = nn.Conv2d(in_channels, widths[0], kernel_size=1)
        self.down = nn.ModuleList()
        channels = widths[0] + embedding_dim
        for w in widths[:-1]:
            self.down.append(DownBlock(channels, w, block_depth))
            channels = w
        self.bottleneck = nn.Sequential(*[
            ResidualBlock(channels if i == 0 else widths[-1], widths[-1]) for i in range(block_depth)
        ])
        channels = widths[-1]
        self.up = nn.ModuleList()
        for w in reversed(widths[:-1]):
            self.up.append(UpBlock(channels, w, w, block_depth))
            channels = w
        self.output_conv = nn.Conv2d(channels, in_channels, kernel_size=1)
        nn.init.zeros_(self.output_conv.weight)  # start by predicting zero noise
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x, t):
        emb = self.time_embedding(t)[:, :, None, None]           # (B, 32, 1, 1)
        emb = self.embedding_upsample(emb)                         # (B, 32, 32, 32)
        h = torch.cat([self.input_conv(x), emb], dim=1)            # (B, 64, 32, 32)
        skips = []
        for block in self.down:
            h = block(h, skips)
        h = self.bottleneck(h)
        for block in self.up:
            h = block(h, skips)
        return self.output_conv(h)


def cosine_alpha_bar(timesteps=1000, s=0.008):
    """Cumulative signal rate alpha_bar_t of the cosine noise schedule (Nichol & Dhariwal, 2021)."""
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0).float()  # (timesteps,), from ~1 down to ~0


class DiffusionModel(nn.Module):
    """DDPM training objective around a U-Net denoiser.

    Forward (noising) process, in closed form:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)
    Training: predict eps from (x_t, t) and minimize MSE(eps_pred, eps).
    """

    def __init__(self, unet, timesteps=1000):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps
        self.register_buffer("alpha_bar", cosine_alpha_bar(timesteps))

    def add_noise(self, x0, t, noise):
        a = self.alpha_bar[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def noise_prediction(self, x0):
        """Sample one random timestep per image, corrupt the images, and predict the noise.
        Returns (predicted_noise, true_noise) for the loss function."""
        t = torch.randint(0, self.timesteps, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.add_noise(x0, t, noise)
        return self.unet(x_t, t), noise

    def training_loss(self, x0):
        predicted, noise = self.noise_prediction(x0)
        return nn.functional.mse_loss(predicted, noise)

    def forward(self, x_t, t):
        return self.unet(x_t, t)


_MODELS = {"FCNN": FCNN, "CNN": CNN, "EnhancedCNN": EnhancedCNN, "VAE": VAE, "GAN": GAN,
           "EBM": EnergyModel, "Diffusion": DiffusionModel}


def get_model(model_name, num_classes=10):
    if model_name not in _MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {list(_MODELS)}")
    if model_name == "VAE":
        return VAE()
    if model_name == "GAN":
        model = GAN()
        model.apply(init_gan_weights)
        return model
    if model_name == "EBM":
        return EnergyModel(in_channels=3)
    if model_name == "Diffusion":
        return DiffusionModel(UNet(image_size=32, in_channels=3), timesteps=1000)
    model = _MODELS[model_name](num_classes=num_classes)
    return model
