"""Model definitions. get_model() returns one of: FCNN, CNN, EnhancedCNN."""
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


_MODELS = {"FCNN": FCNN, "CNN": CNN, "EnhancedCNN": EnhancedCNN}


def get_model(model_name, num_classes=10):
    if model_name not in _MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {list(_MODELS)}")
    model = _MODELS[model_name](num_classes=num_classes)
    return model
