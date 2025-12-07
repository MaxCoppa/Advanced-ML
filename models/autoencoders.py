import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, in_dim, hidden, out_dim, dropout: float = 0.0, batch_norm: bool = False
    ):
        super().__init__()

        layers = []
        last_dim = in_dim
        for h in hidden:

            layers.append(nn.Linear(last_dim, h))

            if batch_norm:
                layers.append(nn.BatchNorm1d(h))

            layers.append(nn.LeakyReLU())

            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

            last_dim = h

        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(
        self, n_feat, hidden, n_latent, dropout: float = 0.0, batch_norm: bool = False
    ):
        super().__init__()
        self.net = MLP(n_feat, hidden, n_latent, dropout, batch_norm)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(
        self, n_latent, hidden, n_recon, dropout: float = 0.0, batch_norm: bool = False
    ):
        super().__init__()
        self.net = MLP(n_latent, hidden, n_recon, dropout, batch_norm)

    def forward(self, z):
        return self.net(z)


class TaskHead(nn.Module):
    def __init__(
        self,
        n_latent,
        hidden,
        n_head_out,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.net = MLP(n_latent, hidden, n_head_out, dropout, batch_norm)

    def forward(self, z):
        return self.net(z)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        n_feat,
        n_latent,
        encoder_hidden,
        decoder_hidden,
        head_hidden,
        dropouts=(0, 0, 0),
        batch_norms=(False, False, False),
    ):
        super().__init__()

        self.encoder = Encoder(
            n_feat, encoder_hidden, n_latent, dropouts[0], batch_norms[0]
        )
        self.decoder = Decoder(
            n_latent, decoder_hidden, n_feat, dropouts[1], batch_norms[1]
        )
        self.head = TaskHead(n_latent, head_hidden, 1, dropouts[2], batch_norms[2])

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        y_hat = self.head(z)
        return z, x_hat, y_hat
