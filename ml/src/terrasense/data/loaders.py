from dataclasses import dataclass

from torch.utils.data import DataLoader, Dataset

from terrasense.config import TrainConfig
from terrasense.data.dataset import load_eurosat
from terrasense.data.splits import load_splits
from terrasense.data.stats import ChannelStats
from terrasense.data.transforms import build_transforms


class TransformSubset(Dataset):
    """Subset of an untransformed dataset with its own transform pipeline."""

    def __init__(self, base: Dataset, indices: list[int], transform):
        self.base = base
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        image, label = self.base[self.indices[i]]
        return self.transform(image), label


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    stats: ChannelStats


def build_dataloaders(cfg: TrainConfig) -> DataBundle:
    processed = cfg.data_dir / "processed"
    if not (processed / "splits.json").exists():
        raise FileNotFoundError(
            f"No splits found in {processed}. Run `python -m terrasense.data.prepare` first."
        )
    splits = load_splits(processed / "splits.json")
    stats = ChannelStats.load(processed / "stats.json")
    base = load_eurosat(cfg.data_dir / "raw", download=False)

    # one base dataset shared by all splits, each split gets its own transform
    def loader(split: str, train: bool) -> DataLoader:
        subset = TransformSubset(base, splits[split], build_transforms(stats, train, cfg.image_size))
        return DataLoader(
            subset,
            batch_size=cfg.batch_size,
            shuffle=train,
            num_workers=cfg.num_workers,
            # trên Windows spawn worker rất chậm, giữ worker sống giữa các epoch
            persistent_workers=cfg.num_workers > 0,
        )

    return DataBundle(
        train=loader("train", True),
        val=loader("val", False),
        test=loader("test", False),
        stats=stats,
    )
