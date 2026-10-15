"""Data loading for CIFAR-10."""
import torch
from torchvision import datasets, transforms

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
# Per-channel mean/std of the CIFAR-10 training set
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transform(train: bool = False, image_size: int = 64):
    """CIFAR-10 images are 32x32; the assignment CNN expects 64x64x3, so we resize."""
    steps = []
    if train:  # light augmentation on the native 32x32 image
        steps += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    steps += [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    return transforms.Compose(steps)


def get_data_loader(data_dir, batch_size=32, train=True, image_size=64, download=True, num_workers=2):
    dataset = datasets.CIFAR10(
        root=data_dir, train=train, download=download,
        transform=get_transform(train=train, image_size=image_size),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers,
    )
    return loader
