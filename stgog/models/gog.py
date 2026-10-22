from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_conv import GCN, chain_adjacency


def compute_num_segments(seq_len: int) -> int:
    """Adaptive heuristic to select segment count based on sequence length."""
    if seq_len < 100:
        return 4
    elif seq_len < 200:
        return 6
    elif seq_len < 400:
        return 8
    else:
        return 10


class TemporalAttentionPooling(nn.Module):
    """
    Learned attention-based pooling over temporal dimension:
      a = Softmax(tanh(W h + b) v)
      out = 0.5 * sum(a * h) + 0.5 * mean(h)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, max(1, dim // 4)),
            nn.Tanh(),
            nn.Linear(max(1, dim // 4), 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        weights = F.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return 0.5 * (x * weights).sum(dim=1) + 0.5 * x.mean(dim=1)


class MultiHeadTemporalAttentionPooling(nn.Module):
    """
    Multi-Head Attention Pooling over sequence dimension.
    Uses multiple attention projections to capture distinct temporal motifs simultaneously.
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Parameter(torch.randn(num_heads, self.head_dim) / math.sqrt(self.head_dim))
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)  # (B, T, H, d_h)
        k = k.permute(0, 2, 1, 3)  # (B, H, T, d_h)

        # Dot product with learned query vector per head
        q = self.q_proj.unsqueeze(0).unsqueeze(2)  # (1, H, 1, d_h)
        scores = (k * q).sum(dim=-1) / math.sqrt(self.head_dim)  # (B, H, T)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, H, T, 1)

        # Weighted sum per head
        pooled_heads = (k * weights).sum(dim=2)  # (B, H, d_h)
        pooled = pooled_heads.reshape(B, D)     # (B, D)
        return self.out_proj(pooled)


class HierarchicalGoG(nn.Module):
    """
    Hierarchical Graph-of-Graphs (GoG) Module.
    1. Partitions sequence into S segments.
    2. Runs Inner GCN (radius r_inner=3) on each sub-series segment.
    3. Pools each segment to form S super-nodes.
    4. Runs Outer GCN (radius r_outer=1) across the S super-nodes.
    5. Readout pools the outer graph to produce a graph-level vector.
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_segments: Optional[int] = None,
        inner_radius: int = 3,
        outer_radius: int = 1,
        pooling: str = "mean",  # 'mean', 'attention', or 'multihead'
    ):
        super().__init__()
        self.num_segments = num_segments
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.pooling_type = pooling

        self.inner_gcn = GCN(in_features, hidden_dim, num_layers=3)
        self.outer_gcn = GCN(hidden_dim, hidden_dim, num_layers=3)

        if pooling == "attention":
            self.inner_pool = TemporalAttentionPooling(hidden_dim)
            self.outer_pool = TemporalAttentionPooling(hidden_dim)
        elif pooling == "multihead":
            self.inner_pool = MultiHeadTemporalAttentionPooling(hidden_dim, num_heads=4)
            self.outer_pool = MultiHeadTemporalAttentionPooling(hidden_dim, num_heads=4)
        else:
            self.inner_pool = None
            self.outer_pool = None

    def _pool(self, x: torch.Tensor, pool_mod: Optional[nn.Module]) -> torch.Tensor:
        if pool_mod is not None:
            return pool_mod(x)
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor, num_segments: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, F_in)
            num_segments: Optional dynamic override for number of segments
        Returns:
            Tensor of shape (B, hidden_dim)
        """
        B, T, _ = x.shape
        device = x.device

        # Determine segment count
        n_seg = num_segments or self.num_segments or compute_num_segments(T)
        n_seg = min(n_seg, T)  # Cannot have more segments than time steps

        # Chunk sequence along temporal dimension (dim=1)
        chunks = torch.chunk(x, n_seg, dim=1)
        segment_reps = []

        for chunk in chunks:
            seg_len = chunk.size(1)
            A_inner = chain_adjacency(seg_len, self.inner_radius, device=device)
            h_inner = self.inner_gcn(chunk, A_inner)
            pooled_seg = self._pool(h_inner, self.inner_pool)
            segment_reps.append(pooled_seg)

        # Stack into super-node graph: (B, S, hidden_dim)
        super_nodes = torch.stack(segment_reps, dim=1)

        # Outer GCN across the segments
        actual_s = super_nodes.size(1)
        A_outer = chain_adjacency(actual_s, self.outer_radius, device=device)
        h_outer = self.outer_gcn(super_nodes, A_outer)

        # Readout: pool across all segments to produce final graph vector (B, hidden_dim)
        out = self._pool(h_outer, self.outer_pool)
        return out
