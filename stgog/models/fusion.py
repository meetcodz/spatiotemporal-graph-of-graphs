from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class PathwayFusion(nn.Module):
    """
    Learnable softmax-weighted fusion across multiple feature pathways.
    Concatenates weighted pathways into a unified representation.
    """
    def __init__(self, num_pathways: int = 4):
        super().__init__()
        self.num_pathways = num_pathways
        # Initialize uniformly
        self.weights = nn.Parameter(torch.ones(num_pathways) / float(num_pathways))

    def get_normalized_weights(self) -> torch.Tensor:
        """Returns the current softmax-normalized pathway weights."""
        return F.softmax(self.weights, dim=0)

    def forward(self, pathways: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            pathways: List of N tensors each of shape (B, D)
        Returns:
            Concatenated weighted features of shape (B, N * D)
        """
        assert len(pathways) == self.num_pathways, (
            f"Expected {self.num_pathways} pathways, got {len(pathways)}"
        )
        w = F.softmax(self.weights, dim=0)
        weighted = [w[i] * p for i, p in enumerate(pathways)]
        return torch.cat(weighted, dim=1)


class ClassificationHead(nn.Module):
    """
    Standard two-block MLP classification head with BatchNorm, ReLU, and Dropout.
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.42,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        h = self.fc1(x)
        if h.size(0) > 1 or not self.training:
            h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Block 2
        h = self.fc2(h)
        if h.size(0) > 1 or not self.training:
            h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Output logits
        return self.fc3(h)
