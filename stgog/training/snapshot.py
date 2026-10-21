from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.augmentations import test_time_augmentation


def get_base_module(model: nn.Module) -> nn.Module:
    """Reach through DataParallel or DistributedDataParallel wrapper if present."""
    if hasattr(model, "module"):
        return model.module
    return model


class SnapshotManager:
    """
    Manages snapshot model checkpoints captured at cosine restart minima.
    Supports in-memory tracking or saving to disk, and runs ensemble inference.
    """
    def __init__(self, checkpoint_dir: Optional[str] = None):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots: List[Dict[str, Any]] = []

    def add_snapshot(self, model: nn.Module, epoch: int, accuracy: float = 0.0):
        """Clone state dict to CPU memory and optionally persist to disk."""
        base_model = get_base_module(model)
        state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
        self.snapshots.append({
            "epoch": epoch,
            "accuracy": accuracy,
            "state_dict": state,
        })
        if self.checkpoint_dir:
            ckpt_file = self.checkpoint_dir / f"snapshot_epoch_{epoch:03d}.pt"
            torch.save(state, ckpt_file)

    def predict_ensemble(
        self,
        loader: DataLoader,
        ref_model: nn.Module,
        device: torch.device,
        tta_passes: int = 7,
        include_best: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Average predicted softmax probability distributions across all captured snapshots.
        Each snapshot evaluates test samples under multi-pass test-time augmentation.
        """
        all_states = [s["state_dict"] for s in self.snapshots]
        if include_best is not None and len(all_states) == 0:
            all_states.append(include_best)

        if not all_states:
            # Fallback to current model weights if no snapshots were saved
            all_states = [{k: v.cpu().clone() for k, v in get_base_module(ref_model).state_dict().items()}]

        base_model = get_base_module(ref_model)
        snapshot_logits = []

        for state in all_states:
            base_model.load_state_dict(state)
            ref_model.eval()
            batch_logits = []
            with torch.no_grad():
                for xb, _ in loader:
                    xb = xb.to(device)
                    # Run Test-Time Augmentation
                    preds = test_time_augmentation(ref_model, xb, passes=tta_passes)
                    batch_logits.append(preds.cpu())
            snapshot_logits.append(torch.cat(batch_logits, dim=0))

        # Mean across snapshots: (N, num_classes)
        return torch.stack(snapshot_logits).mean(dim=0)
