import torch
from torch import nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    # 2 conv 3x3 + BN rồi mới pool, kiểu VGG thu nhỏ
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """Baseline CNN trained from scratch on 64x64 tiles.

    64 -> 32 -> 16 -> 8 spatial, then global average pooling so the head
    doesn't depend on the input size.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))
