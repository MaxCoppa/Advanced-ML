import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_features, d_model=128, num_heads=4, num_layers=4, d_ff=256):
        super().__init__()

        # remplace l'embedding de mots
        self.input_projection = nn.Linear(n_features, d_model)

        self.pos_encoding = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # prédiction finale y_{t+1}
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x : (batch, seq_len, n_features)
        """
        x = self.input_projection(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # On prend le dernier token pour prédire t+1
        last_token = x[:, -1, :]     # (batch, d_model)

        return self.fc_out(last_token)
