import pytest
import torch
import torch.nn as nn

from stgog.models import (
    ResTCN,
    SqueezeExcitation,
    chain_adjacency,
    clear_adjacency_cache,
    GraphConv,
    GCN,
    MultiScaleGCN,
    HierarchicalGoG,
    PathwayFusion,
    ClassificationHead,
    STGoGModel,
)


def test_squeeze_excitation():
    se = SqueezeExcitation(channels=32, reduction=4)
    x = torch.randn(4, 32, 50)
    out = se(x)
    assert out.shape == (4, 32, 50)


def test_restcn_shapes():
    # Test with SE
    tcn_se = ResTCN(in_channels=6, out_channels=32, num_layers=4, use_se=True)
    x = torch.randn(4, 6, 60)
    out = tcn_se(x)
    assert out.shape == (4, 32, 60)

    # Test without SE
    tcn_no_se = ResTCN(in_channels=6, out_channels=32, num_layers=3, use_se=False)
    out_no_se = tcn_no_se(x)
    assert out_no_se.shape == (4, 32, 60)


def test_chain_adjacency():
    clear_adjacency_cache()
    A = chain_adjacency(length=10, radius=2)
    assert A.shape == (10, 10)
    # Symmetry check
    assert torch.allclose(A, A.T, atol=1e-5)
    # Check diagonals are non-zero (self-loops)
    assert torch.all(torch.diag(A) > 0)


def test_graph_conv_and_multiscale_gcn():
    conv = GraphConv(in_features=16, out_features=32)
    x = torch.randn(2, 20, 16)
    A = chain_adjacency(length=20, radius=1)
    out = conv(x, A)
    assert out.shape == (2, 20, 32)

    msgcn = MultiScaleGCN(in_features=8, hidden_dim=32, radii=[1, 3])
    x_ts = torch.randn(2, 25, 8)
    out_ms = msgcn(x_ts)
    assert out_ms.shape == (2, 25, 32)


def test_hierarchical_gog():
    gog = HierarchicalGoG(in_features=16, hidden_dim=32, num_segments=4)
    x = torch.randn(3, 40, 16)
    out = gog(x)
    assert out.shape == (3, 32)


def test_pathway_fusion_and_classifier():
    fusion = PathwayFusion(num_pathways=4)
    p1 = torch.randn(4, 32)
    p2 = torch.randn(4, 32)
    p3 = torch.randn(4, 32)
    p4 = torch.randn(4, 32)
    fused = fusion([p1, p2, p3, p4])
    assert fused.shape == (4, 128)

    classifier = ClassificationHead(in_features=128, num_classes=5)
    logits = classifier(fused)
    assert logits.shape == (4, 5)


def test_stgog_end_to_end_forward():
    model = STGoGModel(
        in_channels=6,
        seq_len=50,
        num_classes=3,
        tcn_hidden=32,
        gcn_hidden=32,
        gog_hidden=32,
        num_segments=4,
    )
    # Input is (B, T, C)
    x = torch.randn(4, 50, 6)
    logits = model(x)
    assert logits.shape == (4, 3)


def test_stgog_strided_downsampling():
    model = STGoGModel(
        in_channels=4,
        seq_len=200,
        num_classes=2,
        tcn_hidden=16,
        gcn_hidden=16,
        gog_hidden=16,
        strided_downsample=2,
    )
    x = torch.randn(2, 200, 4)
    logits = model(x)
    assert logits.shape == (2, 2)


def test_hybrid_adjacency():
    from stgog.models import DynamicCorrelationAdjacency, HybridAdjacency
    x = torch.randn(2, 30, 8)
    dyn = DynamicCorrelationAdjacency(in_features=8, proj_dim=16)
    A_dyn = dyn(x)
    assert A_dyn.shape == (2, 30, 30)

    # Check symmetry and non-negativity under symmetric normalization D^(-1/2) A D^(-1/2)
    assert torch.allclose(A_dyn, A_dyn.transpose(1, 2), atol=1e-5)
    assert torch.all(A_dyn >= 0)
    # Check self loops
    assert torch.all(torch.diagonal(A_dyn, dim1=-2, dim2=-1) > 0)

    hybrid = HybridAdjacency(in_features=8, initial_beta=0.2)
    A_hyb = hybrid(x, radius=2)
    assert A_hyb.shape == (2, 30, 30)
    assert 0.0 <= hybrid.beta <= 1.0



def test_multihead_temporal_attention_pooling():
    from stgog.models import TemporalAttentionPooling, MultiHeadTemporalAttentionPooling
    # Single-head TAP
    tap = TemporalAttentionPooling(dim=32)
    x = torch.randn(4, 15, 32)
    out_tap = tap(x)
    assert out_tap.shape == (4, 32)

    # Multi-head TAP
    mhap = MultiHeadTemporalAttentionPooling(dim=32, num_heads=4)
    out_mhap = mhap(x)
    assert out_mhap.shape == (4, 32)



def test_stgog_inspection_and_hybrid():
    model = STGoGModel(
        in_channels=5,
        seq_len=40,
        num_classes=3,
        tcn_hidden=16,
        gcn_hidden=16,
        gog_hidden=16,
        use_hybrid_adj=True,
    )
    x = torch.randn(2, 40, 5)
    logits = model(x)
    assert logits.shape == (2, 3)

    pw_weights = model.get_pathway_weights()
    assert len(pw_weights) == 4
    assert abs(sum(pw_weights.values()) - 1.0) < 1e-4

    gcn_weights = model.get_gcn_branch_weights()
    assert len(gcn_weights) == 3
    assert abs(sum(gcn_weights.values()) - 1.0) < 1e-4

