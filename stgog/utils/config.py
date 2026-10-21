from typing import Dict, Any, Optional
from pathlib import Path
import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    # Data settings
    "dataset": "NATOPS",
    "use_synthetic": False,
    "cache_dir": "data/cache",
    "batch_size": 16,
    "num_workers": 0,
    "pin_memory": False,

    # Architecture settings
    "tcn_hidden": 128,
    "gcn_hidden": 128,
    "gog_hidden": 128,
    "num_segments": None,  # None = adaptively compute based on sequence length
    "dropout": 0.42,
    "use_se": True,
    "gcn_radii": [1, 3, 5],
    "pooling": "mean",     # 'mean' or 'attention'
    "strided_downsample": None,

    # Training settings
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "label_smoothing": None,
    "mixup_alpha": 0.2,
    "stochastic_p": 0.2,
    "tta_passes": 7,
    "use_amp": True,
    "seed": 42,
    "print_every": 10,

    # Output paths
    "output_dir": "outputs",
    "checkpoint_dir": "checkpoints",
}


def load_config(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file, merge with DEFAULT_CONFIG, and apply any CLI overrides.
    """
    cfg = DEFAULT_CONFIG.copy()
    if config_path:
        p = Path(config_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
            cfg.update(user_cfg)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v

    return cfg


def save_config(config: Dict[str, Any], save_path: str):
    """Serialize configuration dict to a YAML file."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
