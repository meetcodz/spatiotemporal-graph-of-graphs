import pytest
import numpy as np
import torch

from stgog.data import (
    MTSDataset,
    generate_synthetic_mts,
    create_dataloaders,
    MixUp,
    apply_jitter,
    apply_scaling,
)


def test_synthetic_data_generation():
    X_tr, y_tr, X_te, y_te, nc = generate_synthetic_mts(
        n_train=50, n_test=20, seq_len=30, n_channels=4, num_classes=3
    )
    assert X_tr.shape == (50, 30, 4)
    assert y_tr.shape == (50,)
    assert X_te.shape == (20, 30, 4)
    assert y_te.shape == (20,)
    assert nc == 3
    assert set(np.unique(y_tr)).issubset({0, 1, 2})


def test_dataloaders():
    X_tr, y_tr, X_te, y_te, _ = generate_synthetic_mts(n_train=40, n_test=20, seq_len=20)
    train_loader, test_loader = create_dataloaders(X_tr, y_tr, X_te, y_te, batch_size=8)

    xb, yb = next(iter(train_loader))
    assert xb.shape == (8, 20, 6)
    assert yb.shape == (8,)


def test_mixup():
    mixup = MixUp(alpha=0.2)
    x = torch.randn(8, 20, 4)
    y = torch.tensor([0, 1, 0, 1, 2, 2, 0, 1])
    mixed_x, y_a, y_b, lam = mixup(x, y)

    assert mixed_x.shape == x.shape
    assert y_a.shape == y.shape
    assert y_b.shape == y.shape
    assert 0.0 <= lam <= 1.0


def test_jitter_and_scaling():
    x = torch.ones(4, 10, 3)
    jittered = apply_jitter(x, std=0.01, p=1.0)
    assert jittered.shape == x.shape
    assert not torch.allclose(jittered, x)

    scaled = apply_scaling(x, min_scale=0.9, max_scale=1.1, p=1.0)
    assert scaled.shape == x.shape
