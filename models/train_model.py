# PyTorch core
import torch
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device="cpu",
    n_epochs=60,
):
    model = model.to(device)
    criterion = criterion.to(device)

    r2_rec_val_list = []
    r2_sup_val_list = []
    r2_rec_train_list = []
    r2_sup_train_list = []
    loss_list = []

    for epoch in tqdm(range(n_epochs)):
        model.train()
        total_loss = 0.0

        # Accumulate train predictions
        X_train_all, X_hat_train_all = [], []
        y_train_all, y_hat_train_all = [], []

        # Train
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()

            _, x_hat, y_hat = model(xb)

            loss = criterion(x_hat, xb, y_hat, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            X_train_all.append(xb.detach())
            X_hat_train_all.append(x_hat.detach())
            y_train_all.append(yb.detach())
            y_hat_train_all.append(y_hat.detach())

        X_train_all = torch.cat(X_train_all).cpu().numpy()
        X_hat_train_all = torch.cat(X_hat_train_all).cpu().numpy()
        y_train_all = torch.cat(y_train_all).cpu().numpy()
        y_hat_train_all = torch.cat(y_hat_train_all).cpu().numpy()

        r2_rec_train = r2_score(X_train_all, X_hat_train_all)
        r2_sup_train = r2_score(y_train_all, y_hat_train_all)

        r2_rec_train_list.append(r2_rec_train)
        r2_sup_train_list.append(r2_sup_train)

        # Val
        model.eval()
        with torch.no_grad():
            X_val_all, X_hat_all = [], []
            y_val_all, y_hat_all = [], []

            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                _, x_hat, y_hat = model(xb)

                X_val_all.append(xb)
                X_hat_all.append(x_hat)
                y_val_all.append(yb)
                y_hat_all.append(y_hat)

            X_val_all = torch.cat(X_val_all).cpu().numpy()
            X_hat_all = torch.cat(X_hat_all).cpu().numpy()
            y_val_all = torch.cat(y_val_all).cpu().numpy()
            y_hat_all = torch.cat(y_hat_all).cpu().numpy()

            r2_rec_val = r2_score(X_val_all, X_hat_all)
            r2_sup_val = r2_score(y_val_all, y_hat_all)

        r2_rec_val_list.append(r2_rec_val)
        r2_sup_val_list.append(r2_sup_val)
        loss_list.append(total_loss)

        print(
            f"Epoch {epoch+1:02d} | loss={total_loss:.3f} "
            f"| R2_rec_train={r2_rec_train:.4f} | R2_rec_val={r2_rec_val:.4f} "
            f"| R2_sup_train={r2_sup_train:.4f} | R2_sup_val={r2_sup_val:.4f}"
        )

    # Plot
    plot_r2(r2_rec_train_list, r2_rec_val_list, r2_sup_train_list, r2_sup_val_list)

    return True


def plot_r2(r2_rec_train, r2_rec_val, r2_sup_train, r2_sup_val):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    epochs = range(1, len(r2_rec_train) + 1)

    # Matplotlib aesthetics
    linewidth = 2.0

    # Rec
    ax = axes[0]
    ax.plot(epochs, r2_rec_train, label="Train", color="tab:blue", linewidth=linewidth)
    ax.plot(epochs, r2_rec_val, label="Val", color="tab:orange", linewidth=linewidth)

    # y-axis never below -0.5
    ymin = max(-0.05, min(min(r2_rec_train), min(r2_rec_val)))
    ymax = max(max(r2_rec_train), max(r2_rec_val)) * 1.05
    ax.set_ylim(ymin, ymax)

    ax.set_ylabel("R² (reconstruction)")
    ax.set_title("Reconstruction R²")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Sup
    ax = axes[1]
    ax.plot(epochs, r2_sup_train, label="Train", color="tab:green", linewidth=linewidth)
    ax.plot(epochs, r2_sup_val, label="Val", color="tab:red", linewidth=linewidth)

    ymin = max(-0.5, min(min(r2_sup_train), min(r2_sup_val)))
    ymax = max(max(r2_sup_train), max(r2_sup_val)) * 1.05
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("R² (supervision)")
    ax.set_title("Supervision R²")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()
