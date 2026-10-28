"""Model definitions. get_model() returns one of: FCNN, CNN, EnhancedCNN, GAN."""
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


_MODELS = {"FCNN": FCNN, "CNN": CNN, "EnhancedCNN": EnhancedCNN, "GAN": GAN}


def get_model(model_name, num_classes=10):
    if model_name not in _MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {list(_MODELS)}")
    if model_name == "GAN":
        model = GAN()
        model.apply(init_gan_weights)
        return model
    model = _MODELS[model_name](num_classes=num_classes)
    return model
