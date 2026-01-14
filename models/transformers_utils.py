import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import polars as pl
import pandas as pd
import numpy as np
from tqdm import tqdm

class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for time-series forecasting.
    Converts various input formats into windowed sequences for sequential models.
    """
    def __init__(self, X, y, seq_len=50):
        # ----------------------------------------------------
        # Input Feature Handling (X)
        # ----------------------------------------------------
        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()
        elif isinstance(X, pl.Series):
            X = X.to_numpy().reshape(-1, 1)
        elif isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        elif isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise TypeError(f"Unsupported type for X: {type(X)}")

        # ----------------------------------------------------
        # Target Handling (y)
        # ----------------------------------------------------
        if isinstance(y, pl.DataFrame):
            y = y.select(y.columns[0]).to_numpy().reshape(-1)
        elif isinstance(y, pl.Series):
            y = y.to_numpy().reshape(-1)
        elif isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy().reshape(-1)
        elif isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy().reshape(-1)
        elif isinstance(y, np.ndarray):
            y = y.reshape(-1)
        else:
            raise TypeError(f"Unsupported type for y: {type(y)}")

        # Store and Validate
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = seq_len

        total_timesteps = len(self.X)

        if len(self.y) != total_timesteps:
            raise ValueError(f"X and y must have the same length: {total_timesteps} vs {len(self.y)}")

        # n_samples accounts for the lookback window and the target offset
        self.n_samples = total_timesteps - seq_len
        if self.n_samples <= 0:
            raise ValueError("Insufficient data for the chosen sequence length.")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Slice window from X and take the target at the next time step
        x_window = self.X[idx : idx + self.seq_len]
        y_target = self.y[idx + self.seq_len - 1] # Or idx + seq_len depending on your lag logic

        return torch.tensor(x_window), torch.tensor([y_target])

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
    plot=False
):
    """
    Full training loop for sequential models with real-time R2 score tracking.
    """
    model.to(device)

    # Initialize Datasets and Loaders
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_len)
    val_dataset = TimeSeriesDataset(X_val, y_val, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    r2_train_hist = []
    r2_val_hist = []

    if plot:
        plt.ion()  # Interactive mode ON

    for epoch in tqdm(range(epochs)):
        # ======== TRAINING PHASE ========
        model.train()
        y_true_train = []
        y_pred_train = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

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

        # ======== VALIDATION PHASE ========
        model.eval()
        y_true_val = []
        y_pred_val = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                y_true_val.append(yb.cpu())
                y_pred_val.append(preds.cpu())

        y_true_val = torch.cat(y_true_val).numpy()
        y_pred_val = torch.cat(y_pred_val).numpy()
        r2_val = r2_score(y_true_val, y_pred_val)
        r2_val_hist.append(r2_val)

        # ======== LOGGING & REAL-TIME PLOTTING ========
        print(
            f"Epoch {epoch+1:03d} | "
            f"R² Train: {r2_train:.4f} | "
            f"R² Val: {r2_val:.4f}"
        )

        if plot:
            plt.clf()
            plt.plot(r2_train_hist, label="Train R²", color='blue')
            plt.plot(r2_val_hist, label="Val R²", color='orange')
            plt.xlabel("Epochs")
            plt.ylabel("R² Score")
            plt.title("Model Performance during Training")
            plt.legend()
            plt.grid(True)
            plt.pause(0.01)

    if plot:
        plt.ioff() 
        plt.show()

    return r2_train_hist, r2_val_hist