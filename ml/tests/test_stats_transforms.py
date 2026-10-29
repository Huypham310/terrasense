import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from terrasense.data.stats import ChannelStats, compute_channel_stats
from terrasense.data.transforms import build_transforms


class SolidColor(Dataset):
    """Images filled with one known color, so mean/std have an exact answer."""

    def __init__(self, colors):
        self.colors = colors

    def __len__(self):
        return len(self.colors)

    def __getitem__(self, i):
        array = np.full((8, 8, 3), self.colors[i], dtype=np.uint8)
        return Image.fromarray(array), 0


def test_channel_stats_match_numpy():
    ds = SolidColor([(0, 128, 255), (255, 128, 0)])
    stats = compute_channel_stats(ds, indices=[0, 1], batch_size=1)
    # R channel is half 0 and half 255 -> mean 0.5, std 0.5
    assert stats.mean[0] == pytest.approx(0.5, abs=1e-6)
    assert stats.std[0] == pytest.approx(0.5, abs=1e-6)
    # G is constant -> zero variance
    assert stats.mean[1] == pytest.approx(128 / 255, abs=1e-6)
    assert stats.std[1] == pytest.approx(0.0, abs=1e-6)


def test_stats_only_use_given_indices():
    ds = SolidColor([(0, 0, 0), (255, 255, 255)])
    stats = compute_channel_stats(ds, indices=[0])
    assert stats.mean == pytest.approx([0.0, 0.0, 0.0])


def test_stats_save_load(tmp_path):
    stats = ChannelStats(mean=[0.1, 0.2, 0.3], std=[0.4, 0.5, 0.6])
    stats.save(tmp_path / "stats.json")
    assert ChannelStats.load(tmp_path / "stats.json") == stats


@pytest.mark.parametrize("train", [True, False])
def test_transforms_output_shape_and_dtype(train, fake_tiles):
    stats = ChannelStats(mean=[0.5] * 3, std=[0.25] * 3)
    image, _ = fake_tiles[0]
    out = build_transforms(stats, train=train)(image)
    assert out.shape == (3, 64, 64)
    assert out.dtype == torch.float32


def test_eval_transform_normalizes():
    stats = ChannelStats(mean=[0.5] * 3, std=[0.5] * 3)
    white = Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8))
    out = build_transforms(stats, train=False)(white)
    # (1.0 - 0.5) / 0.5 = 1
    assert torch.allclose(out, torch.ones_like(out))


def test_eval_transform_is_deterministic(fake_tiles):
    stats = ChannelStats(mean=[0.5] * 3, std=[0.25] * 3)
    tf = build_transforms(stats, train=False)
    image, _ = fake_tiles[3]
    assert torch.equal(tf(image), tf(image))


def test_resize_when_image_size_differs(fake_tiles):
    stats = ChannelStats(mean=[0.5] * 3, std=[0.25] * 3)
    image, _ = fake_tiles[0]
    out = build_transforms(stats, train=False, image_size=32)(image)
    assert out.shape == (3, 32, 32)
