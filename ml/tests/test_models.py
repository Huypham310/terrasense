import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from terrasense.models import MODEL_NAMES, build_model
from terrasense.train import run_epoch


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_shape(name):
    # pretrained=False để test không phải tải weights ImageNet
    model = build_model(name, num_classes=10, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(4, 3, 64, 64))
    assert out.shape == (4, 10)


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("vgg99", num_classes=10)


def test_simple_cnn_handles_other_input_sizes():
    # global average pooling means the head doesn't care about spatial size
    model = build_model("simple_cnn", num_classes=10).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 96, 96)).shape == (2, 10)


def _toy_loader():
    # 2 classes that are trivially separable by mean brightness
    x = torch.cat([torch.full((32, 3, 16, 16), -1.0), torch.full((32, 3, 16, 16), 1.0)])
    x += 0.1 * torch.randn_like(x)
    y = torch.cat([torch.zeros(32, dtype=torch.long), torch.ones(32, dtype=torch.long)])
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)


def test_run_epoch_learns_toy_problem():
    model = build_model("simple_cnn", num_classes=2)
    loader = _toy_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    first_loss, _ = run_epoch(model, loader, criterion, optimizer)
    for _ in range(8):
        last_loss, _ = run_epoch(model, loader, criterion, optimizer)
    # so sánh loss train với train thôi, eval dùng BN running stats nên không cùng thang
    assert last_loss < first_loss

    _, eval_acc = run_epoch(model, loader, criterion)
    assert eval_acc >= 0.9


def test_run_epoch_eval_does_not_update_weights():
    model = build_model("simple_cnn", num_classes=2)
    before = [p.clone() for p in model.parameters()]
    run_epoch(model, _toy_loader(), nn.CrossEntropyLoss())
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


def test_run_epoch_limit_batches():
    model = build_model("simple_cnn", num_classes=2)
    _, acc = run_epoch(model, _toy_loader(), nn.CrossEntropyLoss(), limit_batches=1)
    # chỉ 1 batch 16 ảnh nên acc phải là bội số của 1/16
    assert (acc * 16) == pytest.approx(round(acc * 16))
