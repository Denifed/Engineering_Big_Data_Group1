"""Simple CNN baseline with window-average horizontal velocity supervision."""
import torch.nn as nn


class CNN1D(nn.Module):
    """输入 (B,6,W)，输出整个窗口的平均 [vx,vy]，单位 m/s。"""

    def __init__(self, in_channels=6, output_dim=2):
        super().__init__()
        self.in_channels = in_channels

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.in_channels or x.shape[2] < 4:
            raise ValueError(f"Expected [B,{self.in_channels},W>=4], got {tuple(x.shape)}")
        x = self.features(x)
        return self.regressor(x)
