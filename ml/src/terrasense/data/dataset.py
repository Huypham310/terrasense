from pathlib import Path

from torchvision.datasets import EuroSAT

from terrasense.config import CLASS_NAMES


def load_eurosat(root: Path, download: bool = True) -> EuroSAT:
    """Load EuroSAT RGB (27,000 64x64 Sentinel-2 tiles) without transforms.

    Transforms are applied per split later so train and eval can use different pipelines.
    """
    root.mkdir(parents=True, exist_ok=True)
    dataset = EuroSAT(root=str(root), download=download)
    if dataset.classes != CLASS_NAMES:
        raise ValueError(f"Unexpected class list: {dataset.classes}")
    return dataset
