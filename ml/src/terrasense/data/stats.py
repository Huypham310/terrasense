import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import v2


@dataclass
class ChannelStats:
    mean: list[float]
    std: list[float]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ChannelStats":
        return cls(**json.loads(path.read_text()))


class _ToTensorDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> torch.Tensor:
        image, _ = self.base[index]
        return self.to_tensor(image)


def compute_channel_stats(dataset: Dataset, indices: list[int], batch_size: int = 512) -> ChannelStats:
    """Per-channel mean/std over the training split only, to avoid leaking val/test statistics."""
    loader = DataLoader(_ToTensorDataset(Subset(dataset, indices)), batch_size=batch_size)
    total = torch.zeros(3, dtype=torch.float64)
    total_sq = torch.zeros(3, dtype=torch.float64)
    pixels = 0
    for batch in loader:
        total += batch.sum(dim=(0, 2, 3), dtype=torch.float64)
        total_sq += (batch.double() ** 2).sum(dim=(0, 2, 3))
        pixels += batch.shape[0] * batch.shape[2] * batch.shape[3]
    mean = total / pixels
    std = (total_sq / pixels - mean**2).sqrt()
    return ChannelStats(mean=mean.tolist(), std=std.tolist())
