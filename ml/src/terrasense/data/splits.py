import json
from pathlib import Path

from sklearn.model_selection import train_test_split


def make_splits(
    targets: list[int], val_fraction: float, test_fraction: float, seed: int
) -> dict[str, list[int]]:
    """Stratified train/val/test split so every class keeps the same proportion in each split."""
    indices = list(range(len(targets)))
    # cắt test ra trước, phần còn lại mới chia train/val
    train_val, test = train_test_split(
        indices, test_size=test_fraction, stratify=targets, random_state=seed
    )
    train_val_targets = [targets[i] for i in train_val]
    # val_fraction is relative to the full dataset, so rescale it to the remaining pool
    relative_val = val_fraction / (1 - test_fraction)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val_targets, random_state=seed
    )
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def save_splits(splits: dict[str, list[int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits))


def load_splits(path: Path) -> dict[str, list[int]]:
    return json.loads(path.read_text())
