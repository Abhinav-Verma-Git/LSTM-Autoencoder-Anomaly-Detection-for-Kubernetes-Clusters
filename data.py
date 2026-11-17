"""Data pipeline for LSTM-Autoencoder anomaly detection on Kubernetes network flows.

Follows Svyrydov & Sievierinov (2026):
  1. mean imputation of missing values (Eqs. 5-9), means computed on NORMAL training data
  2. z-score standardisation (Eqs. 10-15), statistics computed on NORMAL training data
  3. sliding windows of T consecutive records (T = 10 in the paper)

Dataset: Sever & Dogan (2023), "A Kubernetes dataset for misuse detection",
https://github.com/yigitsever/kubernetes-dataset (CICFlowMeter flows, label 0 = benign, 1..10 = attack scenarios).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ID_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp", "Label"]

SCENARIO_NAMES = {
    0: "Benign",
    1: "CVE-2020-13379 (Grafana SSRF)",
    2: "Node-RED reconnaissance",
    3: "Node-RED RCE (reverse shell)",
    4: "Node-RED container escape (attack pods)",
    5: "CVE-2021-43798 (Grafana path traversal)",
    6: "CVE-2019-20933 (InfluxDB auth bypass)",
    7: "CVE-2021-30465 (runc symlink race)",
    8: "CVE-2021-25741 (kubelet subPath)",
    9: "CVE-2022-23648 (containerd)",
    10: "CVE-2019-5736 (runc overwrite)",
}


@dataclass
class Capture:
    name: str
    label: int          # 0 benign, >0 attack scenario id
    frame: pd.DataFrame  # time-sorted rows


def load_captures(root: str) -> list[Capture]:
    caps = []
    for f in sorted(glob.glob(os.path.join(root, "benign", "*.csv")) + glob.glob(os.path.join(root, "malicious", "*.csv"))):
        d = pd.read_csv(f)
        d["_ts"] = pd.to_datetime(d["Timestamp"], format="ISO8601")
        d = d.sort_values("_ts", kind="stable").drop(columns="_ts").reset_index(drop=True)
        labels = d["Label"].unique()
        assert len(labels) == 1, f"{f} mixes labels {labels}"
        caps.append(Capture(os.path.basename(f).split(".pcap")[0], int(labels[0]), d))
    return caps


def feature_columns(caps: list[Capture]) -> list[str]:
    cols = [c for c in caps[0].frame.columns if c not in ID_COLS]
    return cols


def to_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    x[~np.isfinite(x)] = np.nan  # inf -> missing, then imputed
    return x


def make_windows(x: np.ndarray, T: int, stride: int = 1) -> np.ndarray:
    if len(x) < T:
        return np.empty((0, T, x.shape[1]), dtype=x.dtype)
    idx = np.arange(0, len(x) - T + 1, stride)
    return np.stack([x[i:i + T] for i in idx])


@dataclass
class Preprocessor:
    """Imputer (Eq. 8) + standard scaler (Eq. 14), fitted on normal training rows only."""
    mu_imp: np.ndarray = None
    mu: np.ndarray = None
    sigma: np.ndarray = None
    keep: np.ndarray = None
    log1p: bool = False

    def fit(self, x: np.ndarray):
        if self.log1p:
            x = np.sign(x) * np.log1p(np.abs(x))
        self.mu_imp = np.nanmean(x, axis=0)
        self.mu_imp = np.where(np.isnan(self.mu_imp), 0.0, self.mu_imp)
        xi = np.where(np.isnan(x), self.mu_imp, x)
        self.mu = xi.mean(axis=0)
        self.sigma = xi.std(axis=0)
        self.keep = self.sigma > 1e-12  # drop features that are constant in normal training data
        self.sigma = np.where(self.keep, self.sigma, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.log1p:
            x = np.sign(x) * np.log1p(np.abs(x))
        xi = np.where(np.isnan(x), self.mu_imp, x)
        z = (xi - self.mu) / self.sigma
        return z[:, self.keep].astype(np.float32)


@dataclass
class Splits:
    X_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    s_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    s_test: np.ndarray
    n_features: int
    info: dict = field(default_factory=dict)


def build_splits(root: str, T: int = 10, train_frac: float = 0.6, val_frac: float = 0.2,
                 attack_split: str = "blocked", log1p: bool = False) -> Splits:
    """Splits are made on raw rows of each capture BEFORE windowing, and windows are formed only inside
    contiguous segments, so no window straddles two splits (no leakage through overlapping windows).

    Benign captures (both protocols): first 60% rows -> train, next 20% -> validation, last 20% -> test.
    Attack captures (never used for training):
      attack_split="blocked"  (Protocol A, paper-comparable): consecutive blocks of max(20, 2T) rows,
                               alternately assigned to validation and test.
      attack_split="chrono"   (Protocol B, strict): first half of the capture -> validation, second half -> test.
    """
    caps = load_captures(root)
    cols = feature_columns(caps)

    parts = {"train": [], "val": [], "test": []}
    for c in caps:
        x = to_matrix(c.frame, cols)
        n = len(x)
        if c.label == 0:
            a, b = int(n * train_frac), int(n * (train_frac + val_frac))
            parts["train"].append((x[:a], c.label))
            parts["val"].append((x[a:b], c.label))
            parts["test"].append((x[b:], c.label))
        elif attack_split == "chrono":
            a = n // 2
            parts["val"].append((x[:a], c.label))
            parts["test"].append((x[a:], c.label))
        elif attack_split == "blocked":
            B = max(20, 2 * T)
            for k, i in enumerate(range(0, n, B)):
                parts["val" if k % 2 == 0 else "test"].append((x[i:i + B], c.label))
        else:
            raise ValueError(attack_split)

    pre = Preprocessor(log1p=log1p).fit(np.concatenate([p[0] for p in parts["train"]]))

    def win(split):
        Xs, ss = [], []
        for x, lab in parts[split]:
            w = make_windows(pre.transform(x), T)
            Xs.append(w)
            ss.append(np.full(len(w), lab))
        X = np.concatenate(Xs).astype(np.float32)
        s = np.concatenate(ss)
        return X, (s > 0).astype(int), s

    X_train, _, _ = win("train")
    X_val, y_val, s_val = win("val")
    X_test, y_test, s_test = win("test")
    info = {
        "n_raw_features": len(cols),
        "n_features_used": int(pre.keep.sum()),
        "dropped_constant_features": [c for c, k in zip(cols, pre.keep) if not k],
        "window": T,
        "attack_split": attack_split,
        "n_train_windows": int(len(X_train)),
        "n_val_windows": {"normal": int((y_val == 0).sum()), "attack": int((y_val == 1).sum())},
        "n_test_windows": {"normal": int((y_test == 0).sum()), "attack": int((y_test == 1).sum())},
        "raw_rows": {"benign": int(sum(len(c.frame) for c in caps if c.label == 0)),
                     "attack": int(sum(len(c.frame) for c in caps if c.label > 0))},
        "missing_or_inf_cells": int(sum(np.isnan(to_matrix(c.frame, cols)).sum() for c in caps)),
    }
    return Splits(X_train, X_val, y_val, s_val, X_test, y_test, s_test, int(pre.keep.sum()), info)


# ---------------------------------------------------------------------------------------------------
# Flow-level evaluation (v2): every flow gets exactly one decision.
# ---------------------------------------------------------------------------------------------------
@dataclass
class FlowSplits:
    X_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray      # 1 if the window's newest flow is an attack flow
    s_val: np.ndarray      # scenario id of the newest flow (0 = benign)
    c_val: np.ndarray      # scenario id of the capture the flow came from (0 = benign capture)
    a_val: np.ndarray      # 1 if ANY flow in the window is an attack flow (sensitivity analysis)
    X_test: np.ndarray
    y_test: np.ndarray
    s_test: np.ndarray
    c_test: np.ndarray
    a_test: np.ndarray
    n_features: int
    info: dict = field(default_factory=dict)


def causal_windows(z: np.ndarray, T: int) -> np.ndarray:
    """Window i = flows i-T+1 .. i of the same capture; the capture's first flow is repeated as left padding."""
    if T > 1:
        z = np.concatenate([np.repeat(z[:1], T - 1, axis=0), z])
    w = np.lib.stride_tricks.sliding_window_view(z, (T, z.shape[1]))[:, 0]
    return w


def build_flow_splits(root: str, T: int = 10, attack_split: str = "blocked", labels: str = "flow",
                      log1p: bool = False, train_frac: float = 0.6, val_frac: float = 0.2,
                      block: int = 20) -> FlowSplits:
    """Flow-level splits.

    labels="flow"    : attack captures relabelled per flow with relabel.flow_labels (R1/R2 rules)
    labels="capture" : original dataset labels (every flow of an attack capture is an attack)

    Benign captures: first 60% of flows -> train, next 20% -> validation, last 20% -> test (chronological).
    Attack captures (never used for training):
      "blocked" (Protocol A): consecutive blocks of `block` flows alternately -> validation / test
      "chrono"  (Protocol B): first half -> validation, second half -> test
    Each evaluated flow is scored with the causal window ending at it; the block size is fixed, so the set
    of evaluated flows is identical for every window length T.
    """
    from relabel import flow_labels

    caps = load_captures(root)
    cols = feature_columns(caps)
    benign_train_rows = []
    per_cap = []
    for c in caps:
        x = to_matrix(c.frame, cols)
        n = len(x)
        if c.label == 0:
            lab = np.zeros(n, int)
            a, b = int(n * train_frac), int(n * (train_frac + val_frac))
            split = np.array(["train"] * a + ["val"] * (b - a) + ["test"] * (n - b))
            benign_train_rows.append(x[:a])
        else:
            lab = flow_labels(c.frame, c.label)[0] if labels == "flow" else np.full(n, c.label)
            if attack_split == "blocked":
                split = np.array(["val" if (i // block) % 2 == 0 else "test" for i in range(n)])
            elif attack_split == "chrono":
                split = np.array(["val"] * (n // 2) + ["test"] * (n - n // 2))
            else:
                raise ValueError(attack_split)
        per_cap.append((x, lab, split, c.label))

    pre = Preprocessor(log1p=log1p).fit(np.concatenate(benign_train_rows))
    out = {k: {"X": [], "s": [], "c": [], "a": []} for k in ["train", "val", "test"]}
    for x, lab, split, caplab in per_cap:
        W = causal_windows(pre.transform(x), T)
        is_att = (lab > 0).astype(np.int8)
        any_att = (np.lib.stride_tricks.sliding_window_view(
            np.concatenate([np.zeros(T - 1, np.int8), is_att]), T).max(axis=1) if T > 1 else is_att)
        for k in ["train", "val", "test"]:
            m = split == k
            if m.any():
                out[k]["X"].append(W[m].astype(np.float32))
                out[k]["s"].append(lab[m])
                out[k]["c"].append(np.full(m.sum(), caplab))
                out[k]["a"].append(any_att[m])
    cat = lambda k, f: np.concatenate(out[k][f])
    X_tr = cat("train", "X")
    S = FlowSplits(X_tr,
                   cat("val", "X"), (cat("val", "s") > 0).astype(int), cat("val", "s"), cat("val", "c"), cat("val", "a").astype(int),
                   cat("test", "X"), (cat("test", "s") > 0).astype(int), cat("test", "s"), cat("test", "c"), cat("test", "a").astype(int),
                   int(pre.keep.sum()))
    def counts(y, s, c):
        return {"normal": int((y == 0).sum()), "attack": int((y == 1).sum()),
                "normal_from_attack_captures": int(((y == 0) & (c > 0)).sum()),
                "attack_by_scenario": {int(k): int((s == k).sum()) for k in np.unique(s) if k > 0}}
    S.info = {"labels": labels, "attack_split": attack_split, "window": T, "log1p": log1p,
              "n_raw_features": len(cols), "n_features_used": int(pre.keep.sum()),
              "n_train_windows": int(len(X_tr)),
              "val": counts(S.y_val, S.s_val, S.c_val), "test": counts(S.y_test, S.s_test, S.c_test)}
    return S
