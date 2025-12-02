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

    r2_rec_test_list = []
    r2_sup_test_list = []
    loss_list = []
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        # TRAIN LOOP
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()

            z, x_hat, y_hat = model(xb)

            loss = criterion(x_hat, xb, y_hat, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # VALIDATION LOOP
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

            r2_rec_test = r2_score(X_val_all, X_hat_all)
            r2_sup_test = r2_score(y_val_all, y_hat_all)

        r2_rec_test_list.append(r2_rec_test)
        r2_sup_test_list.append(r2_sup_test)
        loss_list.append(total_loss)

        print(
            f"Epoch {epoch+1:02d} | loss={total_loss:.3f} "
            f"| R2_rec_val={r2_rec_test:.4f} "
            f"| R2_sup_val={r2_sup_test:.4f}"
        )

    # Plot
    plot_r2(r2_rec_test_list, r2_sup_test_list)

    return r2_rec_test_list, r2_sup_test_list, loss_list


def plot_r2(r2_rec_list, r2_sup_list):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- subplot 1 : reconstruction ---
    axes[0].plot(r2_rec_list, color="tab:blue")
    axes[0].set_ylabel("R² (reconstruction)")
    axes[0].set_title("Évolution du R² reconstruction")
    axes[0].grid(True)

    # --- subplot 2 : supervision ---a
    axes[1].plot(r2_sup_list, color="tab:green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("R² (supervision)")
    axes[1].set_title("Évolution du R² supervision")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
