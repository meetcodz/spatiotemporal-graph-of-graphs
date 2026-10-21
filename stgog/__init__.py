"""
ST-GoG: Spatiotemporal Graph-of-Graphs Framework for Multivariate Time Series Classification.
"""

from .models import (
    STGoGModel,
    ResTCN,
    SqueezeExcitation,
    MultiScaleGCN,
    HierarchicalGoG,
    PathwayFusion,
    ClassificationHead,
    chain_adjacency,
)
from .data import (
    MTSDataset,
    load_uea_dataset,
    generate_synthetic_mts,
    create_dataloaders,
    MixUp,
    test_time_augmentation,
)
from .training import (
    STGoGTrainer,
    SnapshotManager,
    build_cosine_scheduler,
)
from .utils import (
    load_config,
    save_config,
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_training_curves,
)

__version__ = "1.0.0"

__all__ = [
    "STGoGModel",
    "ResTCN",
    "SqueezeExcitation",
    "MultiScaleGCN",
    "HierarchicalGoG",
    "PathwayFusion",
    "ClassificationHead",
    "chain_adjacency",
    "MTSDataset",
    "load_uea_dataset",
    "generate_synthetic_mts",
    "create_dataloaders",
    "MixUp",
    "test_time_augmentation",
    "STGoGTrainer",
    "SnapshotManager",
    "build_cosine_scheduler",
    "load_config",
    "save_config",
    "compute_classification_metrics",
    "plot_confusion_matrix",
    "plot_training_curves",
    "__version__",
]
