from typing import Set
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    T_0: int = 50,
    T_mult: int = 2,
    eta_min: float = 1e-6,
) -> CosineAnnealingWarmRestarts:
    """Create Cosine Annealing scheduler with warm restarts."""
    return CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
    )


def compute_stochastic_depth(initial_p: float, epoch: int, decay_rate: float = 60.0) -> float:
    """Compute exponentially decaying stochastic depth drop probability for current epoch."""
    return float(initial_p * np.exp(-epoch / decay_rate))


def get_snapshot_epochs(total_epochs: int, T_0: int = 50, T_mult: int = 2) -> Set[int]:
    """Calculate the cycle boundary epochs where snapshots should be captured."""
    snapshots = set()
    period = T_0
    current_ep = 0
    while True:
        current_ep += period
        if current_ep > total_epochs:
            break
        snapshots.add(current_ep)
        period *= T_mult
    return snapshots
