import torch
import torch.nn as nn


class CNNGRU(nn.Module):
    """输入 (B, 6, 200)，输出窗口平均水平速度 [vx,vy]，单位 m/s。"""

    def __init__(self, in_channels=6, output_dim=2, hidden_size=64):
        super().__init__()

        self.cnn = nn.Sequential(
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
        )

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        # (B, 6, 200) -> CNN -> (B, 128, 50)
        x = self.cnn(x)

        # GRU 要求 (B, 时间步, 特征维度)
        x = x.transpose(1, 2)  # (B, 50, 128)
        _, hidden = self.gru(x)

        # 最后一层 GRU 的最终隐藏状态
        x = hidden[-1]  # (B, hidden_size)
        return self.regressor(x)
