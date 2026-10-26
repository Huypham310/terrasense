"""Download EuroSAT, create stratified splits, and compute normalization stats.

Usage: python -m terrasense.data.prepare
"""

import argparse
from collections import Counter

from terrasense.config import CLASS_NAMES, TrainConfig
from terrasense.data.dataset import load_eurosat
from terrasense.data.splits import make_splits, save_splits
from terrasense.data.stats import compute_channel_stats


def main() -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-fraction", type=float, default=defaults.val_fraction)
    parser.add_argument("--test-fraction", type=float, default=defaults.test_fraction)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args()

    dataset = load_eurosat(defaults.data_dir / "raw", download=True)
    splits = make_splits(dataset.targets, args.val_fraction, args.test_fraction, args.seed)
    processed = defaults.data_dir / "processed"
    save_splits(splits, processed / "splits.json")

    stats = compute_channel_stats(dataset, splits["train"])
    stats.save(processed / "stats.json")

    print(f"Total images: {len(dataset)}")
    for name, indices in splits.items():
        counts = Counter(dataset.targets[i] for i in indices)
        per_class = ", ".join(f"{CLASS_NAMES[c]}={counts[c]}" for c in sorted(counts))
        print(f"{name:>5}: {len(indices):>6} | {per_class}")
    print(f"mean={[round(m, 4) for m in stats.mean]} std={[round(s, 4) for s in stats.std]}")


if __name__ == "__main__":
    main()
