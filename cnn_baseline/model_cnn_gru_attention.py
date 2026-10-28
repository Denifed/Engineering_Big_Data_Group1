"""CNN-GRU + 加性时间注意力，回归输入窗口的平均水平速度。

默认维度：[B,6,200] -> CNN [B,128,50] -> GRU [B,50,64]
         -> 时间注意力 [B,64] -> 全连接回归 [B,2]。
只有时间汇总方式与 CNN-GRU 对照不同；不改变其 CNN、GRU 或回归头。
"""
import torch
from torch import nn

from .model_cnn_gru import CNNGRU


class TemporalAttention(nn.Module):
    """Learn scores over [B,T,H]; return weighted context [B,H] and [B,T].

    score_t = v^T tanh(W h_t + b), alpha = softmax(score, time).
    All windows are fixed length; no padding mask is needed.
    """
    def __init__(self, hidden_size=64, attention_size=32):
        super().__init__()
        if hidden_size <= 0 or attention_size <= 0:
            raise ValueError("hidden_size and attention_size must be positive")
        self.hidden_size = hidden_size
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1, bias=False),
        )

    def forward(self, sequence):
        if (sequence.ndim != 3 or sequence.shape[-1] != self.hidden_size
                or sequence.shape[1] == 0):
            raise ValueError(f"Expected [B,T>0,{self.hidden_size}], got {tuple(sequence.shape)}")
        weights = torch.softmax(self.scorer(sequence).squeeze(-1), dim=1)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return context, weights


class CNNGRUAttention(CNNGRU):
    """[B,6,200] -> CNN [B,128,50] -> GRU [B,50,64] -> attention -> [B,2].

    Shares the CNN, GRU and regressor definitions with the CNNGRU control.
    Replaces final-hidden-state selection with attention pooling of all GRU
    outputs. Output is mean horizontal velocity over the input window in m/s.
    """
    def __init__(self, in_channels=6, output_dim=2, hidden_size=64, attention_size=32):
        super().__init__(in_channels=in_channels, output_dim=output_dim, hidden_size=hidden_size)
        self.in_channels = in_channels
        self.attention = TemporalAttention(hidden_size, attention_size)

    def model_config(self):
        """Serializable architecture metadata; training/data settings are separate."""
        return dict(architecture="cnn_gru_additive_temporal_attention_v1",
                    in_channels=self.in_channels, output_dim=self.regressor[-1].out_features,
                    hidden_size=self.gru.hidden_size,
                    attention_size=self.attention.scorer[0].out_features,
                    attention_axis="GRU time dimension after CNN downsampling",
                    pooling="softmax-weighted sum of all GRU outputs")

    def forward_with_attention(self, x):
        """Return velocity [B,2] and time weights [B,floor(W/4)]."""
        if x.ndim != 3 or x.shape[1] != self.in_channels or x.shape[2] < 4:
            raise ValueError(f"Expected [B,{self.in_channels},W>=4], got {tuple(x.shape)}")
        sequence, _ = self.gru(self.cnn(x).transpose(1, 2))
        context, weights = self.attention(sequence)
        return self.regressor(context), weights

    def forward(self, x, return_attention=False):
        """Default returns [B,2]; opt-in returns ([B,2], [B,floor(W/4)]).

        Weights refer to downsampled GRU steps, not the 200 raw IMU frames.
        They retain gradients when gradient tracking is enabled.
        """
        prediction, weights = self.forward_with_attention(x)
        return (prediction, weights) if return_attention else prediction
