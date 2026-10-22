from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn

from .restcn import ResTCN
from .graph_conv import MultiScaleGCN
from .gog import HierarchicalGoG, compute_num_segments
from .fusion import PathwayFusion, ClassificationHead


class STGoGModel(nn.Module):
    """
    Spatiotemporal Graph-of-Graphs (ST-GoG) end-to-end architecture.
    Integrates:
      1. ResTCN with Squeeze-and-Excitation channel attention (Temporal stream)
      2. Multi-Scale GCN with chain adjacency at radii r in {1, 3, 5} (Spatial stream)
         (Optionally with dynamic hybrid attention-augmented adjacency)
      3. Dual Hierarchical Graph-of-Graphs (Inner/Outer GCNs on temporal segments)
      4. 4-Pathway learnable softmax fusion
      5. Two-block MLP classification head
    """
    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        num_classes: int,
        tcn_hidden: int = 128,
        gcn_hidden: int = 128,
        gog_hidden: int = 128,
        num_segments: Optional[int] = None,
        dropout: float = 0.42,
        use_se: bool = True,
        gcn_radii: List[int] = [1, 3, 5],
        pooling: str = "mean",
        strided_downsample: Optional[int] = None,
        use_hybrid_adj: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.num_segments = num_segments or compute_num_segments(seq_len)
        self.dropout = dropout
        self.use_hybrid_adj = use_hybrid_adj

        # Optional pre-downsampling for ultra-long sequences (e.g. EigenWorms T=17,984)
        self.strided_downsample = strided_downsample
        if strided_downsample is not None and strided_downsample > 1:
            self.pre_pool = nn.AvgPool1d(kernel_size=strided_downsample, stride=strided_downsample)
            effective_len = seq_len // strided_downsample
        else:
            self.pre_pool = None
            effective_len = seq_len

        self.effective_len = effective_len

        # 1. Temporal Stream: Dilated ResTCN + SE
        self.tcn = ResTCN(
            in_channels=in_channels,
            out_channels=tcn_hidden,
            num_layers=5,
            kernel_size=3,
            dropout=dropout,
            use_se=use_se,
        )

        # 2. Spatial Stream: Multi-Scale GCN (optionally hybrid)
        self.multi_gcn = MultiScaleGCN(
            in_features=in_channels,
            hidden_dim=gcn_hidden,
            radii=gcn_radii,
            num_layers_per_branch=3,
            use_hybrid_adj=use_hybrid_adj,
        )

        # 3. Relational Modules: Hierarchical Graph-of-Graphs
        self.gog_tcn = HierarchicalGoG(
            in_features=tcn_hidden,
            hidden_dim=gog_hidden,
            num_segments=self.num_segments,
            inner_radius=3,
            outer_radius=1,
            pooling=pooling,
        )
        self.gog_gcn = HierarchicalGoG(
            in_features=gcn_hidden,
            hidden_dim=gog_hidden,
            num_segments=self.num_segments,
            inner_radius=3,
            outer_radius=1,
            pooling=pooling,
        )

        # 4. Pathway Fusion
        self.fusion = PathwayFusion(num_pathways=4)

        # 5. Classifier Head
        total_feat_dim = tcn_hidden + gog_hidden + gcn_hidden + gog_hidden
        self.classifier = ClassificationHead(
            in_features=total_feat_dim,
            num_classes=num_classes,
            hidden_dims=[512, 256],
            dropout=dropout,
        )

    def set_stochastic_depth(self, p: float):
        """Update stochastic depth probability across all GCN branches."""
        for branch in self.multi_gcn.branches:
            branch.stochastic_p = p
        self.gog_tcn.inner_gcn.stochastic_p = p
        self.gog_tcn.outer_gcn.stochastic_p = p
        self.gog_gcn.inner_gcn.stochastic_p = p
        self.gog_gcn.outer_gcn.stochastic_p = p

    def get_pathway_weights(self) -> Dict[str, float]:
        """Inspect learned softmax weights of the 4 feature pathways."""
        w = self.fusion.get_normalized_weights().detach().cpu().numpy()
        return {
            "tcn_global_pool": float(w[0]),
            "tcn_gog": float(w[1]),
            "gcn_global_pool": float(w[2]),
            "gcn_gog": float(w[3]),
        }

    def get_gcn_branch_weights(self) -> Dict[str, float]:
        """Inspect learned softmax weights of the multi-scale GCN branches."""
        w = self.multi_gcn.get_fusion_weights().detach().cpu().numpy()
        return {
            f"radius_{r}": float(w[idx])
            for idx, r in enumerate(self.multi_gcn.radii)
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, T, C)
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # Pre-downsampling if active
        if self.pre_pool is not None:
            # (B, T, C) -> (B, C, T) -> pool -> (B, T', C)
            x = self.pre_pool(x.transpose(1, 2)).transpose(1, 2)

        # 1. Temporal encoding: ResTCN requires (B, C, T)
        t = self.tcn(x.transpose(1, 2)).transpose(1, 2)  # (B, T, tcn_hidden)

        # 2. Spatial encoding: MultiScaleGCN operates on (B, T, C)
        g = self.multi_gcn(x)  # (B, T, gcn_hidden)

        # Global pooling pathways
        t_pool = t.mean(dim=1)  # (B, tcn_hidden)
        g_pool = g.mean(dim=1)  # (B, gcn_hidden)

        # Relational GoG pathways
        t_gog = self.gog_tcn(t)  # (B, gog_hidden)
        g_gog = self.gog_gcn(g)  # (B, gog_hidden)

        # 3. Softmax-weighted 4-pathway fusion
        fused = self.fusion([t_pool, t_gog, g_pool, g_gog])

        # 4. Classification logits
        logits = self.classifier(fused)
        return logits

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "STGoGModel":
        """Instantiate model directly from a configuration dictionary."""
        return cls(
            in_channels=cfg["in_channels"],
            seq_len=cfg["seq_len"],
            num_classes=cfg["num_classes"],
            tcn_hidden=cfg.get("tcn_hidden", 128),
            gcn_hidden=cfg.get("gcn_hidden", 128),
            gog_hidden=cfg.get("gog_hidden", 128),
            num_segments=cfg.get("num_segments", None),
            dropout=cfg.get("dropout", 0.42),
            use_se=cfg.get("use_se", True),
            gcn_radii=cfg.get("gcn_radii", [1, 3, 5]),
            pooling=cfg.get("pooling", "mean"),
            strided_downsample=cfg.get("strided_downsample", None),
            use_hybrid_adj=cfg.get("use_hybrid_adj", False),
        )
