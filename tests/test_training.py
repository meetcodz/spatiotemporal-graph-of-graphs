import pytest
import torch

from stgog.models import STGoGModel
from stgog.data import generate_synthetic_mts, create_dataloaders
from stgog.training import STGoGTrainer


def test_trainer_smoke():
    X_tr, y_tr, X_te, y_te, num_classes = generate_synthetic_mts(
        n_train=32, n_test=16, seq_len=30, n_channels=4, num_classes=2
    )
    train_loader, test_loader = create_dataloaders(X_tr, y_tr, X_te, y_te, batch_size=8)

    model = STGoGModel(
        in_channels=4,
        seq_len=30,
        num_classes=num_classes,
        tcn_hidden=16,
        gcn_hidden=16,
        gog_hidden=16,
        num_segments=3,
        gcn_radii=[1, 2],
    )

    trainer = STGoGTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        lr=1e-3,
        tta_passes=2,
        use_amp=False,
    )

    # Train for 2 epochs
    results = trainer.fit(epochs=2, print_every=1, verbose=False)

    assert "single_best_acc" in results
    assert "ensemble_acc" in results
    assert "metrics" in results
    assert 0.0 <= results["single_best_acc"] <= 1.0
    assert 0.0 <= results["ensemble_acc"] <= 1.0
    assert len(results["history"]["train_loss"]) == 2
