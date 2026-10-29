from collections import Counter

from terrasense.data.splits import load_splits, make_splits, save_splits

TARGETS = [i % 10 for i in range(1000)]


def test_splits_cover_everything_once():
    splits = make_splits(TARGETS, val_fraction=0.1, test_fraction=0.1, seed=42)
    all_indices = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_indices) == list(range(len(TARGETS)))


def test_split_sizes():
    splits = make_splits(TARGETS, val_fraction=0.1, test_fraction=0.1, seed=42)
    assert len(splits["test"]) == 100
    assert len(splits["val"]) == 100
    assert len(splits["train"]) == 800


def test_splits_are_stratified():
    splits = make_splits(TARGETS, val_fraction=0.1, test_fraction=0.1, seed=42)
    # mỗi class có đúng 10 ảnh trong val và test
    for name in ("val", "test"):
        counts = Counter(TARGETS[i] for i in splits[name])
        assert set(counts.values()) == {10}


def test_splits_are_deterministic():
    a = make_splits(TARGETS, 0.1, 0.1, seed=7)
    b = make_splits(TARGETS, 0.1, 0.1, seed=7)
    c = make_splits(TARGETS, 0.1, 0.1, seed=8)
    assert a == b
    assert a != c


def test_save_and_load_roundtrip(tmp_path):
    splits = make_splits(TARGETS, 0.1, 0.1, seed=42)
    path = tmp_path / "nested" / "splits.json"
    save_splits(splits, path)
    assert load_splits(path) == splits
