from torch import nn

from terrasense.models.simple_cnn import SimpleCNN
from terrasense.models.transfer import build_resnet18

MODEL_NAMES = ("simple_cnn", "resnet18")


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Factory used by train/evaluate/export so model names stay consistent."""
    if name == "simple_cnn":
        return SimpleCNN(num_classes)
    if name == "resnet18":
        return build_resnet18(num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model '{name}', expected one of {MODEL_NAMES}")
