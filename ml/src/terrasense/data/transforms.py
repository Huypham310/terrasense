import torch
from torchvision.transforms import v2

from terrasense.data.stats import ChannelStats


def build_transforms(stats: ChannelStats, train: bool, image_size: int = 64) -> v2.Compose:
    """Satellite tiles have no canonical orientation, so flips and 90-degree rotations are safe."""
    steps: list[v2.Transform] = [v2.ToImage()]
    if image_size != 64:
        steps.append(v2.Resize(image_size, antialias=True))
    if train:
        steps += [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomChoice([v2.RandomRotation((angle, angle)) for angle in (0, 90, 180, 270)]),
            v2.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    steps += [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=stats.mean, std=stats.std),
    ]
    return v2.Compose(steps)
