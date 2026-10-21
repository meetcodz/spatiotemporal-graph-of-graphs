import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Channel-wise Squeeze-and-Excitation attention block for 1D sequences."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Conv1d(channels, reduced, kernel_size=1)
        self.fc2 = nn.Conv1d(reduced, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        w = F.adaptive_avg_pool1d(x, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ResTCN(nn.Module):
    """Dilated Residual Temporal Convolutional Network with optional SE attention."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.42,
        use_se: bool = True,
        reduction: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_se = use_se

        layers = []
        ch = in_channels
        for i in range(num_layers):
            d = 2 ** i
            pad = d * (kernel_size // 2)
            layers.extend([
                nn.Conv1d(ch, out_channels, kernel_size=kernel_size, padding=pad, dilation=d),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            ch = out_channels
        self.net = nn.Sequential(*layers)

        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        if use_se:
            self.se = SqueezeExcitation(out_channels, reduction=reduction)
        else:
            self.se = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, in_channels, T)
        Returns:
            Tensor of shape (B, out_channels, T)
        """
        residual = self.shortcut(x)
        out = self.net(x)
        if self.use_se:
            out = self.se(out)
        return F.relu(out + residual)
