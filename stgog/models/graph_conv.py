from typing import Dict, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

_adj_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}


def chain_adjacency(length: int, radius: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Construct symmetric normalized adjacency for a chain graph of given length and radius.
    Cached per (length, radius, device_str) to avoid redundant allocation.
    """
    dev_str = str(device) if device is not None else "cpu"
    key = (length, radius, dev_str)
    if key in _adj_cache:
        return _adj_cache[key]

    A = torch.zeros(length, length, device=device)
    for i in range(length):
        for r in range(1, radius + 1):
            w = 1.0 / r
            if i - r >= 0:
                A[i, i - r] = w
            if i + r < length:
                A[i, i + r] = w
    # Add self-loops
    A = A + torch.eye(length, device=device)
    # Symmetric normalization: D^(-1/2) A D^(-1/2)
    deg = A.sum(dim=1).clamp(min=1e-6).pow(-0.5)
    D = torch.diag(deg)
    norm_A = D @ A @ D

    _adj_cache[key] = norm_A
    return norm_A


def clear_adjacency_cache():
    """Clear cached adjacency matrices."""
    _adj_cache.clear()


class GraphConv(nn.Module):
    """
    Single-layer Graph Convolution: H' = ReLU(A @ H @ W)
    Supports batched node feature tensors.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features of shape (B, N, F_in)
            A: Normalized adjacency matrix of shape (N, N)
        Returns:
            Tensor of shape (B, N, F_out)
        """
        # Einsum: A[i, j] * x[b, j, f] -> [b, i, f]
        ax = torch.einsum("ij,bjf->bif", A, x)
        return F.relu(self.fc(ax))


class GCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network with stochastic depth regularization.
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_layers: int = 3,
        stochastic_p: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.layers = nn.ModuleList([
            GraphConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.stochastic_p = stochastic_p

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, N, F_in)
            A: Adjacency matrix (N, N)
        Returns:
            Output features (B, N, hidden_dim)
        """
        h = F.relu(self.input_proj(x))
        for layer in self.layers:
            if self.training and self.stochastic_p > 0.0:
                if torch.rand(1, device=x.device).item() < self.stochastic_p:
                    continue  # Stochastic depth drop
            h = layer(h, A)
        return h


class MultiScaleGCN(nn.Module):
    """
    Multi-Scale Graph Convolutional Network.
    Evaluates parallel GCN branches at multiple chain graph radii
    and blends their representations via learnable softmax fusion.
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        radii: List[int] = [1, 3, 5],
        num_layers_per_branch: int = 3,
    ):
        super().__init__()
        self.radii = radii
        self.branches = nn.ModuleList([
            GCN(in_features, hidden_dim, num_layers=num_layers_per_branch)
            for _ in radii
        ])
        # Learnable fusion parameters (initialized to uniform)
        self.fusion_weights = nn.Parameter(torch.ones(len(radii)) / len(radii))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) where T is temporal sequence length and C is feature dim
        Returns:
            Blended representation of shape (B, T, hidden_dim)
        """
        T = x.size(1)
        device = x.device
        branch_outs = []
        for radius, branch in zip(self.radii, self.branches):
            A = chain_adjacency(T, radius, device=device)
            branch_outs.append(branch(x, A))

        # Softmax fusion across branches
        w = F.softmax(self.fusion_weights, dim=0)
        out = sum(w[i] * branch_outs[i] for i in range(len(self.radii)))
        return out
