from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Measures the alignment between predicted model confidence and empirical accuracy.
    ECE < 0.05 indicates well-calibrated confidence estimates.
    """
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    correctness = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi) if i < n_bins - 1 else (confidences >= lo) & (confidences <= hi)
        bin_count = mask.sum()
        if bin_count > 0:
            bin_acc = correctness[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += np.abs(bin_acc - bin_conf) * (bin_count / total_samples)

    return float(ece)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logits_or_probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    # Per-class accuracy
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        per_class_acc = np.nan_to_num(cm_norm.diagonal(), nan=0.0).tolist()

    res = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
    }

    if logits_or_probs is not None:
        # Softmax if not already normalized
        if np.any(logits_or_probs < 0) or not np.allclose(logits_or_probs.sum(axis=-1), 1.0, atol=1e-2):
            exps = np.exp(logits_or_probs - np.max(logits_or_probs, axis=-1, keepdims=True))
            probs = exps / np.sum(exps, axis=-1, keepdims=True)
        else:
            probs = logits_or_probs
        res["ece"] = compute_ece(probs, y_true)

    return res
