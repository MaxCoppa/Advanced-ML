import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math
import polars as pl 
from sklearn.metrics import r2_score
import pandas as pd

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return attn @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch, seq_len, d = x.size()
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, x, y=None, mask=None):
        if y is None:
            y = x

        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(y))
        V = self.split_heads(self.W_v(y))

        attn = scaled_dot_product_attention(Q, K, V, mask)

        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(x.size(0), x.size(1), self.d_model)

        return self.W_o(attn)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=256):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        attn = self.self_attn(x, mask=mask)
        x = self.norm1(x + self.dropout(attn))

        ffn = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn))

        return x


class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_features, d_model=128, num_heads=4, num_layers=4, d_ff=256):
        super().__init__()

        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        last_token = x[:, -1, :]  # prédiction t+1

        return self.fc_out(last_token)

