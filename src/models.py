"""Small sequence models for one-step-ahead log-variance forecasting."""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, n_feat: int = 3, hidden: int = 32, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):                      # x: (B, L, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class TransformerForecaster(nn.Module):
    """Encoder-only Transformer with learned positional embeddings.
    The representation of the last time step is used for the forecast."""

    def __init__(self, n_feat: int = 3, seq_len: int = 22, d_model: int = 32,
                 nhead: int = 4, layers: int = 2, dim_ff: int = 64, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):                      # x: (B, L, F)
        h = self.inp(x) + self.pos[:, : x.size(1)]
        h = self.enc(h)
        return self.head(self.norm(h[:, -1])).squeeze(-1)


def build_model(name: str, n_feat: int, seq_len: int, **kw) -> nn.Module:
    if name == "lstm":
        return LSTMForecaster(n_feat, **kw)
    if name == "transformer":
        return TransformerForecaster(n_feat, seq_len, **kw)
    raise ValueError(name)
