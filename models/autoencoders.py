import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()

        layers = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h

        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, n_feat, hidden, n_latent):
        super().__init__()
        self.net = MLP(n_feat, hidden, n_latent)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, n_latent, hidden, n_recon):
        super().__init__()
        self.net = MLP(n_latent, hidden, n_recon)

    def forward(self, z):
        return self.net(z)


class TaskHead(nn.Module):
    def __init__(self, n_latent, hidden, n_head_out):
        super().__init__()
        self.net = MLP(n_latent, hidden, n_head_out)

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
    ):
        super().__init__()

        self.encoder = Encoder(n_feat, encoder_hidden, n_latent)
        self.decoder = Decoder(n_latent, decoder_hidden, n_feat)
        self.head = TaskHead(n_latent, head_hidden, 1)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        y_hat = self.head(z)
        return z, x_hat, y_hat
