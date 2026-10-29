import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset


class FakeTiles(Dataset):
    """Tiny stand-in for EuroSAT: random 64x64 RGB PIL images with balanced labels.

    Lets the data/model tests run in CI without downloading the real dataset.
    """

    def __init__(self, n: int = 200, num_classes: int = 10, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.images = [
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)) for _ in range(n)
        ]
        self.targets = [i % num_classes for i in range(n)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index], self.targets[index]


@pytest.fixture
def fake_tiles() -> FakeTiles:
    return FakeTiles()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
