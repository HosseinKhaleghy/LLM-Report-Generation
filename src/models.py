"""
Deep-learning regressor for the GBRT-vs-DL comparison.

`TorchMLPRegressor` is a small multilayer perceptron wrapped in a
scikit-learn-style `fit` / `predict` interface so it can be dropped into the
same cross-validation loop as the gradient-boosted baseline.  It standardises
both inputs and the target internally and uses early stopping on a held-out
split, which matters because only 96 farms are available for training.

Keeping this class in its own module means a pickled model bundle can be
re-loaded by `predict.py` without importing the whole training script.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class _MLP(nn.Module):
    """Plain feed-forward network: Linear -> ReLU -> Dropout blocks."""

    def __init__(self, n_features: int, hidden=(128, 64, 32), dropout=0.15):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # noqa: D102
        return self.net(x).squeeze(-1)


class TorchMLPRegressor:
    """scikit-learn-compatible MLP regressor backed by PyTorch."""

    def __init__(self, hidden=(128, 64, 32), dropout=0.15, lr=1e-3,
                 epochs=500, batch_size=16, weight_decay=1e-4,
                 patience=50, val_fraction=0.15, random_state=42):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

    # -- training ------------------------------------------------------------
    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X).astype(np.float32)
        # Standardise the target as well for a well-conditioned loss surface.
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() or 1.0)
        ys = (y - self.y_mean_) / self.y_std_

        # Internal train/validation split for early stopping.
        n = len(Xs)
        idx = rng.permutation(n)
        n_val = max(1, int(round(self.val_fraction * n)))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        Xtr = torch.tensor(Xs[tr_idx]); ytr = torch.tensor(ys[tr_idx])
        Xva = torch.tensor(Xs[val_idx]); yva = torch.tensor(ys[val_idx])

        self.net_ = _MLP(X.shape[1], self.hidden, self.dropout)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_val, best_state, stale = float("inf"), None, 0
        for _ in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(len(Xtr))
            for s in range(0, len(Xtr), self.batch_size):
                b = perm[s:s + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net_(Xtr[b]), ytr[b])
                loss.backward()
                opt.step()

            self.net_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net_(Xva), yva).item()
            if val_loss < best_val - 1e-5:
                best_val, stale = val_loss, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    # -- inference -----------------------------------------------------------
    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler_.transform(X).astype(np.float32)
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(torch.tensor(Xs)).numpy()
        return out * self.y_std_ + self.y_mean_

    def get_params(self, deep=True):  # sklearn compatibility
        return {k: getattr(self, k) for k in
                ("hidden", "dropout", "lr", "epochs", "batch_size",
                 "weight_decay", "patience", "val_fraction", "random_state")}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
