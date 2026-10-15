"""Tests for the Assignment 2 CNN architecture and the helper library."""
import pytest
import torch
from PIL import Image

from helper_lib.data_loader import CIFAR10_CLASSES, get_transform
from helper_lib.evaluator import confusion_matrix, per_class_accuracy
from helper_lib.model import get_model


def test_cnn_output_shape():
    model = get_model("CNN")
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 10)


def test_cnn_matches_assignment_spec():
    """Every layer produces the shape required by the assignment."""
    model = get_model("CNN")
    x = torch.randn(1, 3, 64, 64)
    expected = [
        (1, 16, 64, 64),  # Conv2d 16 filters, 3x3, stride 1, padding 1
        (1, 16, 64, 64),  # ReLU
        (1, 16, 32, 32),  # MaxPool 2x2, stride 2
        (1, 32, 32, 32),  # Conv2d 32 filters
        (1, 32, 32, 32),  # ReLU
        (1, 32, 16, 16),  # MaxPool
        (1, 8192),        # Flatten
        (1, 100),         # Linear 100
        (1, 100),         # ReLU
        (1, 10),          # Linear 10
    ]
    layers = list(model.features) + list(model.classifier)
    assert len(layers) == len(expected)
    for layer, shape in zip(layers, expected):
        x = layer(x)
        assert tuple(x.shape) == shape, f"{layer} gave {tuple(x.shape)}, expected {shape}"


def test_cnn_layer_settings():
    conv1, _, pool1, conv2, _, pool2 = list(get_model("CNN").features)
    for conv, out_ch in [(conv1, 16), (conv2, 32)]:
        assert conv.out_channels == out_ch
        assert conv.kernel_size == (3, 3)
        assert conv.stride == (1, 1)
        assert conv.padding == (1, 1)
    for pool in (pool1, pool2):
        assert pool.kernel_size == 2 and pool.stride == 2


def test_cnn_parameter_count():
    model = get_model("CNN")
    # conv1 448 + conv2 4,640 + fc1 819,300 + fc2 1,010
    assert sum(p.numel() for p in model.parameters()) == 825_398


@pytest.mark.parametrize("name", ["FCNN", "CNN", "EnhancedCNN"])
def test_all_helper_models(name):
    model = get_model(name)
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 10)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        get_model("ResNet")


def test_transform_resizes_cifar_image_to_64():
    image = Image.new("RGB", (32, 32), (120, 60, 200))
    for train in (True, False):
        assert get_transform(train=train)(image).shape == (3, 64, 64)


def test_confusion_matrix_and_per_class_accuracy():
    labels = torch.tensor([0, 0, 1, 1])
    preds = torch.tensor([0, 1, 1, 1])
    matrix = confusion_matrix(preds, labels, num_classes=2)
    assert matrix.tolist() == [[1, 1], [0, 2]]
    assert per_class_accuracy(matrix) == [50.0, 100.0]


def test_ten_classes():
    assert len(CIFAR10_CLASSES) == 10
