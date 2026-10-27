from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet-18 with ImageNet weights and a new classification head.

    The whole network is fine-tuned; EuroSAT is different enough from ImageNet
    that freezing the backbone hurts accuracy.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    # thay lớp fc 1000 class của ImageNet bằng 10 class EuroSAT
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
