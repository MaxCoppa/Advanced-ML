import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x_hat, x_true, y_hat, y_true):
        loss_rec = F.mse_loss(x_hat, x_true)
        loss_sup = F.mse_loss(y_hat, y_true)
        return self.alpha * loss_rec + self.beta * loss_sup
