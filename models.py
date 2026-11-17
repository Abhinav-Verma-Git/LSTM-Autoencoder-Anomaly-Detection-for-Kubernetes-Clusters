"""Detectors: LSTM Autoencoder (proposed) and baselines (dense AE, PCA, Isolation Forest)."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


class LSTMAutoencoder(nn.Module):
    """Encoder: LSTM(h1) -> LSTM(h2) -> latent (last hidden state)
    Decoder: repeat latent T times -> LSTM(h2) -> LSTM(h1) -> Linear(n_features) per step.
    X_hat = D(E(X))  (Eq. 16)."""

    def __init__(self, n_features: int, h1: int = 64, h2: int = 32):
        super().__init__()
        self.enc1 = nn.LSTM(n_features, h1, batch_first=True)
        self.enc2 = nn.LSTM(h1, h2, batch_first=True)
        self.dec1 = nn.LSTM(h2, h2, batch_first=True)
        self.dec2 = nn.LSTM(h2, h1, batch_first=True)
        self.out = nn.Linear(h1, n_features)

    def forward(self, x):
        T = x.shape[1]
        h, _ = self.enc1(x)
        _, (z, _) = self.enc2(h)           # z: (1, B, h2)
        z = z[-1].unsqueeze(1).repeat(1, T, 1)
        h, _ = self.dec1(z)
        h, _ = self.dec2(h)
        return self.out(h)


class DenseAutoencoder(nn.Module):
    """Non-temporal baseline: same windows, flattened."""

    def __init__(self, n_in: int, latent: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, latent), nn.ReLU(),
            nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, n_in))

    def forward(self, x):
        s = x.shape
        return self.net(x.reshape(s[0], -1)).reshape(s)


def train_ae(model: nn.Module, X_train: np.ndarray, X_val_normal: np.ndarray, *, epochs: int = 40,
             batch: int = 256, lr: float = 1e-3, patience: int = 5, seed: int = 0, log=print):
    """MSE training on normal windows only; early stopping on normal validation loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xt = torch.from_numpy(X_train)
    Xv = torch.from_numpy(X_val_normal)
    best, best_state, bad, hist = np.inf, None, 0, []
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for i in range(0, len(Xt), batch):
            xb = Xt[perm[i:i + batch]]
            opt.zero_grad()
            loss = lossf(model(xb), xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(xb)
        tr = tot / len(Xt)
        va = float(np.mean(reconstruction_error(model, X_val_normal)))
        hist.append({"epoch": ep + 1, "train_loss": tr, "val_loss": va, "sec": time.time() - t0})
        log(f"    ep {ep+1:02d} train {tr:.4f} val {va:.4f} ({time.time()-t0:.1f}s)")
        if va < best - 1e-5:
            best, bad = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def reconstruction_error(model: nn.Module, X: np.ndarray, batch: int = 2048) -> np.ndarray:
    """epsilon_i = 1/(T n) sum_t sum_j (x - x_hat)^2   (Eq. 17)."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch])
        out.append(((model(xb) - xb) ** 2).mean(dim=(1, 2)).numpy())
    return np.concatenate(out)


def pca_scores(X_train, X_eval_list, var: float = 0.95):
    f = lambda X: X.reshape(len(X), -1)
    p = PCA(n_components=var, svd_solver="full").fit(f(X_train))
    res = []
    for X in X_eval_list:
        Z = f(X)
        res.append(((p.inverse_transform(p.transform(Z)) - Z) ** 2).mean(axis=1))
    return res, int(p.n_components_)


def iforest_scores(X_train, X_eval_list, seed: int = 0):
    f = lambda X: X.reshape(len(X), -1)
    m = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(f(X_train))
    return [-m.score_samples(f(X)) for X in X_eval_list]
