from typing import Tuple, List, Callable
import numpy as np
import torch
import torch.nn as nn


class MixUp:
    """
    MixUp data augmentation for multivariate time series.
    Interpolates random pairs of examples and their one-hot target distributions:
      x_mixed = lambda * x_i + (1 - lambda) * x_j
      loss = lambda * loss(pred, y_i) + (1 - lambda) * loss(pred, y_j)
    """
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Args:
            x: Input tensor of shape (B, T, C)
            y: Target label tensor of shape (B,)
        Returns:
            mixed_x, y_a, y_b, lam
        """
        if self.alpha <= 0.0:
            return x, y, y, 1.0

        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        mixed_x = lam * x + (1.0 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, float(lam)


def apply_jitter(x: torch.Tensor, std: float = 0.01, p: float = 0.3) -> torch.Tensor:
    """Add small Gaussian noise with probability p."""
    if torch.rand(1, device=x.device).item() < p:
        return x + std * torch.randn_like(x)
    return x


def apply_scaling(x: torch.Tensor, min_scale: float = 0.9, max_scale: float = 1.1, p: float = 0.3) -> torch.Tensor:
    """Multiply by random scalar with probability p."""
    if torch.rand(1, device=x.device).item() < p:
        scale = torch.empty(1, device=x.device).uniform_(min_scale, max_scale).item()
        return x * scale
    return x


def test_time_augmentation(
    model: nn.Module,
    x: torch.Tensor,
    passes: int = 7,
    jitter_std: float = 0.003,
) -> torch.Tensor:
    """
    Evaluate model under multi-pass Test-Time Augmentation (TTA).
    Averages predictions across original and lightly perturbed copies.
    """
    if passes <= 1:
        return model(x)

    preds = [model(x)]
    for _ in range(passes - 1):
        perturbed = x + torch.randn_like(x) * jitter_std
        preds.append(model(perturbed))
    return torch.stack(preds).mean(dim=0)
