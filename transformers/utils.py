import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import polars as pl
import pandas as pd
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len=50):
        # --------------------------
        # Convertir X vers numpy
        # --------------------------
        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()
        elif isinstance(X, pl.Series):
            X = X.to_numpy().reshape(-1, 1)
        elif isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy()
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        elif isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise TypeError(f"X type non supporté : {type(X)}")

        # --------------------------
        # Convertir y vers numpy
        # --------------------------
        if isinstance(y, pl.DataFrame):
            y = y.select(y.columns[0]).to_numpy().reshape(-1)
        elif isinstance(y, pl.Series):
            y = y.to_numpy().reshape(-1)
        elif isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = y.to_numpy().reshape(-1)
        elif isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy().reshape(-1)
        elif isinstance(y, np.ndarray):
            y = y.reshape(-1)
        else:
            raise TypeError(f"y type non supporté : {type(y)}")

        # --------------------------
        # Stocker et vérifier
        # --------------------------
        self.X = X.astype(float)
        self.y = y.astype(float)
        self.seq_len = seq_len

        T = len(self.X)

        if len(self.y) != T:
            raise ValueError(f"X et y doivent avoir la même longueur : {T} vs {len(self.y)}")

        self.n_samples = T - seq_len - 1
        if self.n_samples <= 0:
            raise ValueError("Pas assez de données pour ce seq_len.")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # ici self.X est toujours un np.ndarray → jamais un DF
        x_window = self.X[idx: idx + self.seq_len]
        y_target = self.y[idx + self.seq_len]

        x_window = torch.tensor(x_window, dtype=torch.float32)
        y_target = torch.tensor([y_target], dtype=torch.float32)

        return x_window, y_target

def train_model(
    model,
    optimizer,
    criterion,
    X_train,
    y_train,
    X_val,
    y_val,
    seq_len=50,
    batch_size=32,
    epochs=50,
    device="cpu",
    plot=True
):

    model.to(device)

    train_dataset = TimeSeriesDataset(X_train, y_train, seq_len)
    val_dataset = TimeSeriesDataset(X_val, y_val, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    r2_train_hist = []
    r2_val_hist = []

    if plot:
        plt.ion()  # mode interactif

    for epoch in range(epochs):
        # ======== TRAIN ========
        model.train()
        y_true_train = []
        y_pred_train = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            y_true_train.append(yb.detach().cpu())
            y_pred_train.append(preds.detach().cpu())

        y_true_train = torch.cat(y_true_train).numpy()
        y_pred_train = torch.cat(y_pred_train).numpy()
        r2_train = r2_score(y_true_train, y_pred_train)
        r2_train_hist.append(r2_train)

        # ======== VALIDATION ========
        model.eval()
        y_true_val = []
        y_pred_val = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                y_true_val.append(yb.cpu())
                y_pred_val.append(preds.cpu())

        y_true_val = torch.cat(y_true_val).numpy()
        y_pred_val = torch.cat(y_pred_val).numpy()
        r2_val = r2_score(y_true_val, y_pred_val)
        r2_val_hist.append(r2_val)

        # ======== LOG + PLOT ========
        print(
            f"Epoch {epoch+1:03d} | "
            f"R² train = {r2_train:.4f} | "
            f"R² val = {r2_val:.4f}"
        )

    if plot:
        plt.clf()
        plt.plot(r2_train_hist, label="Train R²")
        plt.plot(r2_val_hist, label="Val R²")
        plt.xlabel("Epoch")
        plt.ylabel("R²")
        plt.legend()
        plt.grid(True)
        plt.pause(0.01)

    return r2_train_hist, r2_val_hist
