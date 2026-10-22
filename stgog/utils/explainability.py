from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..models.stgog_model import STGoGModel


class STGoGExplainer:
    """
    Model Interpretability and Attribution Explainer for ST-GoG.
    Extracts:
      1. Gradient-based spatiotemporal saliency heatmaps across (Time, Channel).
      2. Learned 4-pathway softmax fusion weights.
      3. Multi-scale GCN branch weights across topological radii.
    """
    def __init__(self, model: STGoGModel):
        self.model = model

    def compute_saliency(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute gradient-based attribution w.r.t. input tensor x.
        Args:
            x: Tensor of shape (1, T, C)
            target_class: Target class index. If None, uses top predicted class.
        Returns:
            Dict containing saliency map, channel importance, temporal importance, and predicted class.
        """
        self.model.eval()
        x_in = x.clone().detach().requires_grad_(True)

        logits = self.model(x_in)  # (1, K)
        probs = torch.softmax(logits, dim=-1)

        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())

        score = logits[0, target_class]
        score.backward()

        # Gradient magnitude attribution: (T, C)
        grad = x_in.grad.squeeze(0).detach().cpu().numpy()  # (T, C)
        x_raw = x.squeeze(0).detach().cpu().numpy()          # (T, C)

        # Gradient * Input attribution
        saliency = np.abs(grad * x_raw)

        # Normalize to [0, 1]
        s_min, s_max = saliency.min(), saliency.max()
        if s_max - s_min > 1e-8:
            saliency_norm = (saliency - s_min) / (s_max - s_min)
        else:
            saliency_norm = np.zeros_like(saliency)

        channel_importance = saliency.sum(axis=0)  # (C,)
        if channel_importance.sum() > 0:
            channel_importance /= channel_importance.sum()

        temporal_importance = saliency.sum(axis=1)  # (T,)
        if temporal_importance.sum() > 0:
            temporal_importance /= temporal_importance.sum()

        pathway_weights = self.model.get_pathway_weights()
        branch_weights = self.model.get_gcn_branch_weights()

        return {
            "predicted_class": target_class,
            "class_probability": float(probs[0, target_class].item()),
            "saliency_map": saliency_norm,
            "channel_importance": channel_importance,
            "temporal_importance": temporal_importance,
            "pathway_weights": pathway_weights,
            "branch_weights": branch_weights,
        }

    def get_attribution_summary(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        top_k_channels: int = 5,
    ) -> Dict[str, Any]:
        """
        Produce a high-level summary dict with top ranking channels, temporal hotspots, and weights.
        """
        explanation = self.compute_saliency(x, target_class=target_class)
        ch_imp = explanation["channel_importance"]
        top_indices = np.argsort(ch_imp)[::-1][:top_k_channels].tolist()

        return {
            "predicted_class": explanation["predicted_class"],
            "class_probability": explanation["class_probability"],
            "top_channels": [{"channel_index": int(idx), "importance": float(ch_imp[idx])} for idx in top_indices],
            "pathway_weights": explanation["pathway_weights"],
            "gcn_branch_weights": explanation["branch_weights"],
        }

    def plot_explanation(
        self,
        x: torch.Tensor,
        explanation: Optional[Dict[str, Any]] = None,
        channel_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = "outputs/explanation.png",
    ):
        """
        Render a 4-panel interpretability diagnosis dashboard.
        """
        if explanation is None:
            explanation = self.compute_saliency(x)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        x_np = x.squeeze(0).detach().cpu().numpy()  # (T, C)
        T, C = x_np.shape
        saliency = explanation["saliency_map"]      # (T, C)
        pred_cls = explanation["predicted_class"]
        pred_name = class_names[pred_cls] if class_names else f"Class {pred_cls}"
        prob = explanation["class_probability"]

        if channel_names is None:
            channel_names = [f"Ch {i+1}" for i in range(C)]

        fig = plt.figure(figsize=(14, 10), dpi=150)
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1.2, 0.9], hspace=0.35, wspace=0.3)

        # Panel 1: Multi-Channel Raw Waveforms
        ax1 = fig.add_subplot(gs[0, :])
        for c in range(min(C, 12)):  # Plot up to 12 channels for clarity
            ax1.plot(x_np[:, c], label=channel_names[c], alpha=0.8, linewidth=1.2)
        ax1.set_title(f"Input Signal Waveforms (Prediction: {pred_name} — Confidence: {prob*100:.1f}%)",
                      fontsize=11, fontweight="bold")
        ax1.set_xlabel("Time Step (t)", fontsize=9)
        ax1.set_ylabel("Normalized Value", fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.5)
        if C <= 8:
            ax1.legend(loc="upper right", ncol=min(C, 4), fontsize=8)

        # Panel 2: Spatiotemporal Attribution Heatmap
        ax2 = fig.add_subplot(gs[1, :])
        im = ax2.imshow(saliency.T, aspect="auto", cmap="magma", interpolation="nearest")
        ax2.set_title("Spatiotemporal Attribution Heatmap (Grad * Input)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Time Step (t)", fontsize=9)
        ax2.set_ylabel("Sensor Channel", fontsize=9)
        ax2.set_yticks(range(min(C, 16)))
        ax2.set_yticklabels([channel_names[i] for i in range(min(C, 16))], fontsize=8)
        fig.colorbar(im, ax=ax2, orientation="horizontal", pad=0.2, shrink=0.6, label="Attribution Intensity")

        # Panel 3: Channel Relative Importance
        ax3 = fig.add_subplot(gs[2, 0])
        ch_imp = explanation["channel_importance"]
        colors_bar = ["#1E3A8A" if val >= np.median(ch_imp) else "#94A3B8" for val in ch_imp[:12]]
        ax3.bar(range(min(C, 12)), ch_imp[:12] * 100, color=colors_bar)
        ax3.set_xticks(range(min(C, 12)))
        ax3.set_xticklabels([channel_names[i] for i in range(min(C, 12))], rotation=45, ha="right", fontsize=8)
        ax3.set_title("Top Sensor Importance (%)", fontsize=10, fontweight="bold")
        ax3.set_ylabel("Attribution Share (%)", fontsize=8)
        ax3.grid(True, axis="y", linestyle="--", alpha=0.5)

        # Panel 4: Learned Pathway Fusion Breakdown
        ax4 = fig.add_subplot(gs[2, 1])
        pw = explanation["pathway_weights"]
        pw_names = ["TCN Global", "TCN + GoG", "GCN Global", "GCN + GoG"]
        pw_vals = [pw["tcn_global_pool"], pw["tcn_gog"], pw["gcn_global_pool"], pw["gcn_gog"]]
        colors_pw = ["#3B82F6", "#8B5CF6", "#0D9488", "#10B981"]
        ax4.bar(pw_names, [v * 100 for v in pw_vals], color=colors_pw)
        ax4.set_title("Learned 4-Pathway Fusion Weights", fontsize=10, fontweight="bold")
        ax4.set_ylabel("Softmax Share (%)", fontsize=8)
        ax4.grid(True, axis="y", linestyle="--", alpha=0.5)
        for idx, v in enumerate(pw_vals):
            ax4.text(idx, v * 100 + 1, f"{v*100:.1f}%", ha="center", fontsize=8, fontweight="bold")

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
        return fig

    # Alias for dashboard plotting
    plot_explanation_dashboard = plot_explanation
