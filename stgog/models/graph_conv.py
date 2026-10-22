from typing import Dict, Tuple, Optional, List
import math
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


class DynamicCorrelationAdjacency(nn.Module):
    """
    Computes a data-driven, self-attentive dynamic adjacency matrix:
      A_dyn = Softmax((X W_q)(X W_k)^T / sqrt(d))
    Made symmetric and with self-loops to ensure numerical stability in GCN message passing.
    """
    def __init__(self, in_features: int, proj_dim: Optional[int] = None):
        super().__init__()
        p_dim = proj_dim or max(8, in_features // 2)
        self.q_proj = nn.Linear(in_features, p_dim, bias=False)
        self.k_proj = nn.Linear(in_features, p_dim, bias=False)
        self.scale = 1.0 / math.sqrt(p_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, F_in)
        Returns:
            Symmetric normalized dynamic adjacency of shape (B, N, N)
        """
        B, N, _ = x.shape
        q = self.q_proj(x)  # (B, N, p_dim)
        k = self.k_proj(x)  # (B, N, p_dim)

        # Pairwise dot-product attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, N, N)
        attn = F.softmax(scores, dim=-1)

        # Symmetrize and add self-loops
        A_sym = 0.5 * (attn + attn.transpose(1, 2))
        I = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)
        A = A_sym + I

        # Normalization: D^(-1/2) A D^(-1/2)
        deg = A.sum(dim=-1).clamp(min=1e-6).pow(-0.5)  # (B, N)
        D = torch.diag_embed(deg)  # (B, N, N)
        return torch.bmm(torch.bmm(D, A), D)


class HybridAdjacency(nn.Module):
    """
    Learned convex combination of topological chain adjacency and data-driven correlation:
      A_hybrid = (1 - beta) * A_chain + beta * A_dynamic
    where beta in [0, 1] is learned via sigmoid activation.
    """
    def __init__(self, in_features: int, initial_beta: float = 0.15):
        super().__init__()
        self.dyn_adj = DynamicCorrelationAdjacency(in_features)
        # Inverse sigmoid for initial_beta
        init_logit = math.log(initial_beta / (1.0 - initial_beta + 1e-7))
        self.beta_param = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

    @property
    def beta(self) -> float:
        return float(torch.sigmoid(self.beta_param).item())

    def forward(self, x: torch.Tensor, radius: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, F_in)
            radius: chain radius for topological prior
        Returns:
            Hybrid adjacency of shape (B, N, N)
        """
        B, N, _ = x.shape
        A_chain = chain_adjacency(N, radius, device=x.device)  # (N, N)
        A_chain = A_chain.unsqueeze(0).expand(B, -1, -1)       # (B, N, N)

        A_dyn = self.dyn_adj(x)                               # (B, N, N)
        b = torch.sigmoid(self.beta_param)
        return (1.0 - b) * A_chain + b * A_dyn


class GraphConv(nn.Module):
    """
    Single-layer Graph Convolution: H' = ReLU(A @ H @ W)
    Supports both static 2D adjacency (N, N) and batched 3D adjacency (B, N, N).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features of shape (B, N, F_in)
            A: Normalized adjacency matrix of shape (N, N) or (B, N, N)
        Returns:
            Tensor of shape (B, N, F_out)
        """
        if A.dim() == 2:
            ax = torch.einsum("ij,bjf->bif", A, x)
        else:
            ax = torch.bmm(A, x)
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
            A: Adjacency matrix (N, N) or (B, N, N)
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
    Optionally supports Hybrid (Topological + Dynamic Attention) Adjacency.
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        radii: List[int] = [1, 3, 5],
        num_layers_per_branch: int = 3,
        use_hybrid_adj: bool = False,
    ):
        super().__init__()
        self.radii = radii
        self.use_hybrid_adj = use_hybrid_adj
        self.branches = nn.ModuleList([
            GCN(in_features, hidden_dim, num_layers=num_layers_per_branch)
            for _ in radii
        ])

        if use_hybrid_adj:
            self.hybrid_modules = nn.ModuleList([
                HybridAdjacency(in_features) for _ in radii
            ])
        else:
            self.hybrid_modules = None

        # Learnable fusion parameters (initialized to uniform)
        self.fusion_weights = nn.Parameter(torch.ones(len(radii)) / len(radii))

    def get_fusion_weights(self) -> torch.Tensor:
        """Returns softmax-normalized branch weights."""
        return F.softmax(self.fusion_weights, dim=0)

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

        for idx, (radius, branch) in enumerate(zip(self.radii, self.branches)):
            if self.use_hybrid_adj and self.hybrid_modules is not None:
                A = self.hybrid_modules[idx](x, radius=radius)
            else:
                A = chain_adjacency(T, radius, device=device)
            branch_outs.append(branch(x, A))

        # Softmax fusion across branches
        w = F.softmax(self.fusion_weights, dim=0)
        out = sum(w[i] * branch_outs[i] for i in range(len(self.radii)))
        return out
