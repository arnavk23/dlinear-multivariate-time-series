import torch
import torch.nn as nn

class DLinear(nn.Module):
    """
    DLinear model for multivariate time series forecasting.
    Decomposes input into trend and seasonal components, applies separate linear layers, and combines outputs.
    """
    def __init__(self, input_dim: int, seq_len: int, pred_len: int):
        super().__init__()
        self.trend_linear = nn.Linear(seq_len, pred_len)
        self.seasonal_linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        trend = self.trend_linear(x.mean(dim=-1))
        seasonal = self.seasonal_linear(x - x.mean(dim=-1, keepdim=True))
        return trend + seasonal
