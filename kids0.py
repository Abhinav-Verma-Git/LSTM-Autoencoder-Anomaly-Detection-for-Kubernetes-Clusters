"""Kube-IDS0 (Morsli et al., 2025) loader and evaluation protocols for the DIRECT replication.

Inputs (from the Kaggle release):
  dvwa_dataset_ml_ready.csv, boa_dataset_ml_ready_frontend_microservice.csv   -> features + label (the files whose
      attack count, 42,705, equals the original article's evaluation set)
  dvwa_merged_not_cleaned.csv, boa_merged_not_cleaned_frontend_microservice.csv -> timestamps (ML-ready rows are an
      order-preserving subset of these; alignment computed by align_ml_ready())

Scenario ids used here (global):
  0 benign | 1 DVWA DoS Slowloris | 2 DVWA DoS Tor's Hammer | 3 DVWA brute force | 4 DVWA SQL injection
  5 Bank-of-Anthos Slowloris | 6 Bank-of-Anthos Tor's Hammer
Sessions = (application, recording day): dvwa-11-18, dvwa-11-19 (SQLi only), boa-10-25, boa-10-31.

Protocols
  F (paper-faithful): scaling statistics over ALL rows (as in the article's Eqs. 10-12); LSTM-AE trained on a random
    80% of normal windows (20% for early stopping); evaluation on the FULL sample (training windows included), with the
    F1-optimal threshold chosen on that same set.
  A (blocked): per session, consecutive 20-flow blocks rotate train,train,train,val,test. Training windows = windows
    whose 10 flows are all normal and all in train blocks. Scaling fitted on normal training flows.
  B (chronological): per session, first 60% of flows -> training period (only fully-normal windows used, attack flows
    discarded), next 20% -> validation, last 20% -> test.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data import Preprocessor, causal_windows

HERE = os.path.dirname(os.path.abspath(__file__))
# Folder holding the Kube-IDS0 CSVs (any sub-folder layout). Override with the KIDS0_DIR environment variable.
ROOT = os.environ.get("KIDS0_DIR", os.path.join(HERE, "data", "kube-ids0"))
# (ML-ready file pattern, timestamped merged file pattern, label -> global scenario id)
FILES = {
    "dvwa": ("dvwa_dataset_ml_ready", "dvwa_merged_not_cleaned", {1: 1, 2: 2, 3: 3, 4: 4}),
    "boa": ("boa_dataset_ml_ready", "boa_merged_not_cleaned", {1: 5, 2: 6}),
}
SCEN = {0: "Benign", 1: "DVWA DoS (Slowloris)", 2: "DVWA DoS (Tor's Hammer)", 3: "DVWA brute force",
        4: "DVWA SQL injection", 5: "BoA DoS (Slowloris)", 6: "BoA DoS (Tor's Hammer)"}
CACHE = os.path.join(ROOT, "kids0_cache.pkl")


def find_csv(pattern: str) -> str:
    """Find exactly one CSV under ROOT whose name contains `pattern` (skips macOS '__MACOSX' / '._' junk files)."""
    hits = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d != "__MACOSX"]
        for f in filenames:
            if f.lower().endswith(".csv") and not f.startswith("._") and pattern in f:
                hits.append(os.path.join(dirpath, f))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected exactly one CSV containing '{pattern}' under {ROOT}, found {len(hits)}: {hits}\n"
            "Put the Kube-IDS0 CSV files (unzipped) in that folder, or set the KIDS0_DIR environment variable.")
    return hits[0]


def align_ml_ready(ml: pd.DataFrame, merged: pd.DataFrame) -> np.ndarray:
    shared = [c for c in ml.columns if c in merged.columns and c != "label"][:40]
    A = np.nan_to_num(np.round(merged[shared].to_numpy(float), 4), nan=-9)
    B = np.nan_to_num(np.round(ml[shared].to_numpy(float), 4), nan=-9)
    j, idx = 0, []
    for i in range(len(B)):
        while j < len(A) and not np.allclose(A[j], B[i], atol=1e-3):
            j += 1
        assert j < len(A), "alignment failed"
        idx.append(j); j += 1
    return np.array(idx)


def load() -> pd.DataFrame:
    """One frame, rows time-ordered within each session; columns = common features + label/scen/session/ts."""
    if os.path.exists(CACHE):
        return pd.read_pickle(CACHE)
    frames, featsets = [], []
    for app, (fml, fmg, lmap) in FILES.items():
        ml = pd.read_csv(find_csv(fml), low_memory=False)
        mg = pd.read_csv(find_csv(fmg), low_memory=False)
        idx = align_ml_ready(ml, mg)
        assert (mg.label.to_numpy()[idx] == ml.label.to_numpy()).all()
        ts = pd.to_datetime(pd.Series(mg.timestamp.to_numpy()[idx]), format="ISO8601")
        extra = pd.DataFrame({"ts": ts.to_numpy(),
                              "scen": ml.label.map(lambda v: 0 if v == 0 else lmap[v]).to_numpy(),
                              "session": (app + "-" + ts.dt.strftime("%m-%d")).to_numpy()})
        ml = pd.concat([ml.reset_index(drop=True), extra], axis=1)
        frames.append(ml); featsets.append(set(ml.columns) - {"label", "ts", "scen", "session"})
    common = sorted(set.intersection(*featsets))
    df = pd.concat([f[common + ["label", "scen", "session", "ts"]] for f in frames], ignore_index=True)
    df = df.sort_values(["session", "ts"], kind="stable").reset_index(drop=True)
    df.to_pickle(CACHE)
    return df


def feature_cols(df: pd.DataFrame, group: str = "all") -> list[str]:
    cols = [c for c in df.columns if c not in ("label", "scen", "session", "ts")]
    if group == "net":
        cols = [c for c in cols if not c.startswith("container_")]
    elif group == "ctr":
        cols = [c for c in cols if c.startswith("container_")]
    return cols


@dataclass
class KSplits:
    X_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    s_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    s_test: np.ndarray
    conc_test: np.ndarray   # 1 for a normal test flow recorded inside an attack interval of its own session
    sess_test: np.ndarray
    a_test: np.ndarray      # any-flow window label
    n_features: int
    info: dict = field(default_factory=dict)


def _attack_intervals(df):
    iv = {}
    for (sess, s), g in df[df.scen > 0].groupby(["session", "scen"]):
        iv.setdefault(sess, []).append((g.ts.min(), g.ts.max()))
    return iv


def build(protocol: str, T: int = 10, group: str = "all", seed: int = 0, block: int = 20) -> KSplits:
    df = load()
    cols = feature_cols(df, group)
    iv = _attack_intervals(df)
    conc = np.zeros(len(df), bool)
    for sess, ivs in iv.items():
        m = (df.session == sess).to_numpy()
        for a, b in ivs:
            conc |= m & (df.ts >= a).to_numpy() & (df.ts <= b).to_numpy()
    conc &= (df.scen == 0).to_numpy()

    split = np.empty(len(df), dtype=object)
    for sess, g in df.groupby("session", sort=False):
        ix = g.index.to_numpy(); n = len(ix)
        if protocol == "F":
            split[ix] = "eval"
        elif protocol == "A":
            pat = np.array(["train", "train", "train", "val", "test"])
            split[ix] = pat[(np.arange(n) // block) % 5]
        elif protocol == "B":
            a, b = int(0.6 * n), int(0.8 * n)
            split[ix] = np.array(["train"] * a + ["val"] * (b - a) + ["test"] * (n - b))
        else:
            raise ValueError(protocol)

    x = df[cols].to_numpy(np.float64)
    x[~np.isfinite(x)] = np.nan
    lab = df.scen.to_numpy()
    is_tr = split == "train"
    fit_rows = x if protocol == "F" else x[is_tr & (lab == 0)]
    pre = Preprocessor().fit(fit_rows)

    W_all, ok_train, any_att = [], [], []
    for sess, g in df.groupby("session", sort=False):
        ix = g.index.to_numpy()
        W_all.append((ix, causal_windows(pre.transform(x[ix]), T)))
        att = (lab[ix] > 0).astype(np.int8); trn = is_tr[ix].astype(np.int8)
        pad = lambda v, fill: np.concatenate([np.full(T - 1, fill, np.int8), v])
        sw = np.lib.stride_tricks.sliding_window_view
        if T > 1:
            any_att.append((ix, sw(pad(att, 0), T).max(1)))
            # padded positions repeat the first flow: treat them as that flow
            att_p = np.concatenate([np.repeat(att[:1], T - 1), att]); trn_p = np.concatenate([np.repeat(trn[:1], T - 1), trn])
            ok_train.append((ix, (sw(att_p, T).max(1) == 0) & (sw(trn_p, T).min(1) == 1)))
        else:
            any_att.append((ix, att)); ok_train.append((ix, (att == 0) & (trn == 1)))
    W = np.empty((len(df), T, int(pre.keep.sum())), np.float32)
    okt = np.zeros(len(df), bool); anyl = np.zeros(len(df), np.int8)
    for (ix, w), (_, o), (_, a) in zip(W_all, ok_train, any_att):
        W[ix] = w; okt[ix] = o; anyl[ix] = a

    y = (lab > 0).astype(int)
    if protocol == "F":
        normal_ix = np.where(y == 0)[0]
        rng = np.random.default_rng(seed); rng.shuffle(normal_ix)
        k = int(0.8 * len(normal_ix))
        tr_ix, es_ix = np.sort(normal_ix[:k]), np.sort(normal_ix[k:])
        ev = np.arange(len(df))
        S = KSplits(W[tr_ix], W[es_ix], y[es_ix], lab[es_ix], W[ev], y[ev], lab[ev], conc[ev], df.session.to_numpy()[ev],
                    anyl[ev], int(pre.keep.sum()))
    else:
        tr = np.where(okt & is_tr)[0]; va = np.where(split == "val")[0]; te = np.where(split == "test")[0]
        S = KSplits(W[tr], W[va], y[va], lab[va], W[te], y[te], lab[te], conc[te], df.session.to_numpy()[te], anyl[te],
                    int(pre.keep.sum()))

    def cnt(yy, ss):
        return {"normal": int((yy == 0).sum()), "attack": int((yy == 1).sum()),
                "by_scenario": {int(k): int((ss == k).sum()) for k in np.unique(ss) if k > 0}}
    S.info = {"protocol": protocol, "window": T, "features": group, "n_features_used": int(pre.keep.sum()),
              "n_features_raw": len(cols), "n_train_windows": int(len(S.X_train)),
              "val": cnt(S.y_val, S.s_val), "test": cnt(S.y_test, S.s_test),
              "test_concurrent_normal": int(S.conc_test.sum()),
              "scaling_fit": "all rows" if protocol == "F" else "normal training flows",
              "rows_total": int(len(df)), "attack_rows_total": int((lab > 0).sum())}
    return S
