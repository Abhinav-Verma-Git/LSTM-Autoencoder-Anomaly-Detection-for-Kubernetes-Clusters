"""Experiments on flow-level relabelled data (v2). Writes results_v2/*.json and score files.

  python run_v2.py --part main      # LSTM-AE, Dense-AE, IForest (5 seeds) + PCA; Protocols A and B; T=10
  python run_v2.py --part capture   # same windowing, ORIGINAL capture-level labels; Protocol A; 3 seeds
  python run_v2.py --part ablation  # LSTM-AE: T in {1,5,20}, log1p features; Protocol A; 3 seeds
  python run_v2.py --part extra     # LSTM-AE: up to 200 epochs (patience 10), mean vs max-step score; 3 seeds
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data import SCENARIO_NAMES, build_flow_splits
from metrics import evaluate, f1_optimal_threshold, percentile_threshold
from models import (DenseAutoencoder, LSTMAutoencoder, iforest_scores, pca_scores, reconstruction_error,
                    train_ae)

torch.set_num_threads(os.cpu_count() or 2)
OUT = "results_v2"
os.makedirs(OUT, exist_ok=True)
DATA = "data_k8s"


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(os.path.join(OUT, "run.log"), "a") as f:
        f.write(msg + "\n")


def score_both(name, seed, sv, st, S, extra=None):
    th_f1 = f1_optimal_threshold(sv, S.y_val)
    th_p99 = percentile_threshold(sv[S.y_val == 0], 99.0)
    r = {"model": name, "seed": seed,
         "val_at_f1_threshold": evaluate(sv, S.y_val, th_f1),
         "test_f1_threshold": evaluate(st, S.y_test, th_f1, S.s_test, S.c_test),
         "test_p99_threshold": evaluate(st, S.y_test, th_p99, S.s_test, S.c_test),
         # sensitivity: window labelled attack if ANY of its flows is an attack flow
         "test_roc_auc_anyflow_label": float(roc_auc_score(S.a_test, st))}
    if extra:
        r.update(extra)
    a, b = r["test_f1_threshold"], r["test_p99_threshold"]
    log(f"  RESULT {name:14s} seed {seed}: ROC {a['roc_auc']:.4f} PR {a['pr_auc']:.4f} expl {a['roc_auc_excl_recon']:.4f} "
        f"recon {a['roc_auc_recon']:.4f} | F1thr P {a['precision_attack']:.3f} R {a['recall_attack']:.3f} "
        f"F1 {a['f1_attack']:.3f} FPR {a['fpr']:.3f} | p99 P {b['precision_attack']:.3f} R {b['recall_attack']:.3f} "
        f"FPR {b['fpr']:.3f}")
    return r


def run_ae(kind, S, seed, epochs=60, patience=5, label=None):
    m = LSTMAutoencoder(S.n_features) if kind == "LSTM-AE" else DenseAutoencoder(S.X_train.shape[1] * S.n_features)
    n_params = sum(p.numel() for p in m.parameters())
    t0 = time.time()
    m, hist = train_ae(m, S.X_train, S.X_val[S.y_val == 0], epochs=epochs, patience=patience, seed=seed, log=log)
    train_sec = time.time() - t0
    t1 = time.time()
    sv, st = reconstruction_error(m, S.X_val), reconstruction_error(m, S.X_test)
    infer = (time.time() - t1) / (len(S.X_val) + len(S.X_test)) * 1000
    r = score_both(label or kind, seed, sv, st, S, {"n_params": int(n_params), "train_sec": train_sec,
                                                    "epochs_run": len(hist), "history": hist,
                                                    "inference_ms_per_window": infer})
    return r, sv, st, m


def save(name, obj):
    json.dump(obj, open(os.path.join(OUT, name), "w"), indent=1, default=float)


def part_main(seeds):
    for proto, split in [("A", "blocked"), ("B", "chrono")]:
        log(f"===== MAIN PROTOCOL {proto} ({split}) =====")
        S = build_flow_splits(DATA, T=10, attack_split=split, labels="flow")
        log("DATA", json.dumps(S.info))
        runs, saved = [], {}
        for seed in seeds:
            for kind in ["LSTM-AE", "Dense-AE"]:
                log(f"Training {kind} seed {seed}")
                r, sv, st, _ = run_ae(kind, S, seed)
                runs.append(r)
                if seed == seeds[0]:
                    saved[kind] = (sv, st)
            t0 = time.time()
            sv, st = iforest_scores(S.X_train, [S.X_val, S.X_test], seed=seed)
            runs.append(score_both("IForest", seed, sv, st, S, {"train_sec": time.time() - t0}))
            if seed == seeds[0]:
                saved["IForest"] = (sv, st)
        t0 = time.time()
        (sv, st), k = pca_scores(S.X_train, [S.X_val, S.X_test])
        runs.append(score_both("PCA", 0, sv, st, S, {"n_components": k, "train_sec": time.time() - t0}))
        saved["PCA"] = (sv, st)
        np.savez_compressed(os.path.join(OUT, f"scores_seed0_{proto}.npz"), y_val=S.y_val, s_val=S.s_val,
                            c_val=S.c_val, y_test=S.y_test, s_test=S.s_test, c_test=S.c_test, a_test=S.a_test,
                            **{f"{k}__val": v[0] for k, v in saved.items()},
                            **{f"{k}__test": v[1] for k, v in saved.items()})
        save(f"main_{proto}.json", {"protocol": proto, "data": S.info, "scenarios": SCENARIO_NAMES, "runs": runs})


def part_capture(seeds):
    log("===== CAPTURE-LEVEL LABELS, PROTOCOL A =====")
    S = build_flow_splits(DATA, T=10, attack_split="blocked", labels="capture")
    log("DATA", json.dumps(S.info))
    runs = [run_ae("LSTM-AE", S, s, label="LSTM-AE(capture-labels)")[0] for s in seeds]
    save("capture_labels.json", {"data": S.info, "runs": runs})


def part_ablation(seeds):
    runs = []
    for T in [1, 5, 20]:
        S = build_flow_splits(DATA, T=T, attack_split="blocked", labels="flow")
        log(f"===== ABLATION T={T} =====")
        for s in seeds:
            r = run_ae("LSTM-AE", S, s, label=f"LSTM-AE(T={T})")[0]; r["T"] = T; r["log1p"] = False; runs.append(r)
    S = build_flow_splits(DATA, T=10, attack_split="blocked", labels="flow", log1p=True)
    log("===== ABLATION log1p T=10 =====")
    for s in seeds:
        r = run_ae("LSTM-AE", S, s, label="LSTM-AE(log1p)")[0]; r["T"] = 10; r["log1p"] = True; runs.append(r)
    save("ablation.json", {"runs": runs})


@torch.no_grad()
def step_errors(model, X, batch=2048):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch])
        out.append(((model(xb) - xb) ** 2).mean(dim=2).numpy())
    return np.concatenate(out)


def part_extra(seeds):
    S = build_flow_splits(DATA, T=10, attack_split="blocked", labels="flow")
    runs = []
    for s in seeds:
        log(f"===== EXTRA long training seed {s} =====")
        m, hist = train_ae(LSTMAutoencoder(S.n_features), S.X_train, S.X_val[S.y_val == 0], epochs=200,
                           patience=10, seed=s, log=log)
        ev, et = step_errors(m, S.X_val), step_errors(m, S.X_test)
        for agg in ["mean", "max", "last"]:
            f = {"mean": lambda e: e.mean(1), "max": lambda e: e.max(1), "last": lambda e: e[:, -1]}[agg]
            r = score_both(f"LSTM-AE(200ep,{agg})", s, f(ev), f(et), S, {"epochs_run": len(hist)})
            r["score"] = agg; runs.append(r)
    save("extra.json", {"runs": runs})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["main", "capture", "ablation", "extra"], required=True)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    {"main": part_main, "capture": part_capture, "ablation": part_ablation, "extra": part_extra}[a.part](list(range(a.seeds)))
    log("DONE", a.part)
