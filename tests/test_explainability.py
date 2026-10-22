import pytest
import torch
import numpy as np
from pathlib import Path

from stgog.models import STGoGModel
from stgog.utils.explainability import STGoGExplainer


def test_explainer_saliency_and_summary(tmp_path):
    model = STGoGModel(
        in_channels=4,
        seq_len=30,
        num_classes=3,
        tcn_hidden=16,
        gcn_hidden=16,
        gog_hidden=16,
        num_segments=3,
    )
    explainer = STGoGExplainer(model)

    x = torch.randn(1, 30, 4)
    res = explainer.compute_saliency(x)

    assert "saliency_map" in res
    assert "channel_importance" in res
    assert "temporal_importance" in res
    assert "predicted_class" in res
    assert "class_probability" in res

    assert res["saliency_map"].shape == (30, 4)
    assert res["channel_importance"].shape == (4,)
    assert res["temporal_importance"].shape == (30,)
    assert 0 <= res["predicted_class"] < 3
    assert 0.0 <= res["class_probability"] <= 1.0

    # Test summary
    summary = explainer.get_attribution_summary(x, top_k_channels=2)
    assert len(summary["top_channels"]) == 2
    assert "pathway_weights" in summary
    assert "gcn_branch_weights" in summary

    # Test dashboard plot generation
    save_path = tmp_path / "dashboard.png"
    fig = explainer.plot_explanation_dashboard(x, save_path=save_path)
    assert save_path.exists()
