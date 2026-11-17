"""Direct replication on Kube-IDS0. Writes results_kids0/*.json and per-seed score files.

  python run_kids0.py --part main      # Protocols F, A, B: LSTM-AE, Dense-AE, IForest (5 seeds) + PCA
  python run_kids0.py --part ablation  # LSTM-AE: Protocol A with T=1, network-only, container-only; F network-only
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import kids0
from metrics import evaluate, f1_optimal_threshold, percentile_threshold
from models import (DenseAutoencoder, LSTMAutoencoder, iforest_scores, pca_scores, reconstruction_error,
                    train_ae)

torch.set_num_threads(os.cpu_count() or 2)
OUT = "results_kids0"
os.makedirs(os.path.join(OUT, "scores"), exist_ok=True)


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(os.path.join(OUT, "run.log"), "a") as f:
        f.write(msg + "\n")


def auc(neg, pos, e):
    if neg.sum() == 0 or pos.sum() == 0:
        return None
    return float(roc_auc_score(np.r_[np.zeros(neg.sum()), np.ones(pos.sum())], np.r_[e[neg], e[pos]]))


def extra_metrics(st, S, thr):
    y, s, c = S.y_test, S.s_test, S.conc_test
    conc_att = (y == 1) & (s != 4)              # attacks that have benign traffic recorded at the same time
    out = {
        "auc_attack_vs_concurrent_normal": auc(c, conc_att, st),
        "auc_attack_vs_nonconcurrent_normal": auc((y == 0) & ~c, conc_att, st),
        "auc_concurrent_vs_nonconcurrent_normal": auc((y == 0) & ~c, c, st),
        "per_scenario_auc": {int(k): auc(y == 0, s == k, st) for k in np.unique(s) if k > 0},
        "per_scenario_auc_vs_concurrent": {int(k): auc(c, s == k, st) for k in np.unique(s) if k > 0 and k != 4},
        "roc_auc_anyflow_label": float(roc_auc_score(S.a_test, st)),
    }
    for name, th in thr.items():
        yhat = st >= th
        out[f"fpr_concurrent_{name}"] = float(yhat[c].mean()) if c.any() else None
        out[f"fpr_nonconcurrent_{name}"] = float(yhat[(y == 0) & ~c].mean()) if ((y == 0) & ~c).any() else None
    return out


def score(name, seed, sv, st, S, protocol, extra=None):
    if protocol == "F":
        # as described in the article: threshold maximising F1, chosen on the (full) evaluation set
        thr = {"f1": f1_optimal_threshold(st, S.y_test)}
        thr["p99"] = percentile_threshold(sv, 99.0)  # sv = held-out NORMAL windows
    else:
        thr = {"f1": f1_optimal_threshold(sv, S.y_val), "p99": percentile_threshold(sv[S.y_val == 0], 99.0)}
    r = {"model": name, "seed": seed, "protocol": protocol,
         "test_f1_threshold": evaluate(st, S.y_test, thr["f1"], S.s_test),
         "test_p99_threshold": evaluate(st, S.y_test, thr["p99"], S.s_test),
         "extra": extra_metrics(st, S, thr)}
    for k in ["roc_auc_excl_recon", "roc_auc_recon", "recall_exploits", "recall_recon"]:  # Sever-Dogan-specific
        r["test_f1_threshold"].pop(k, None); r["test_p99_threshold"].pop(k, None)
    if extra:
        r.update(extra)
    a, b, x = r["test_f1_threshold"], r["test_p99_threshold"], r["extra"]
    log(f"  RESULT {protocol} {name:16s} seed {seed}: ROC {a['roc_auc']:.4f} PR {a['pr_auc']:.4f} | F1thr P {a['precision_attack']:.3f} "
        f"R {a['recall_attack']:.3f} F1 {a['f1_attack']:.3f} Rn {a['recall_normal']:.3f} Acc {a['accuracy']:.3f} | p99 P {b['precision_attack']:.3f} "
        f"R {b['recall_attack']:.3f} FPR {b['fpr']:.3f} | conc {x['auc_attack_vs_concurrent_normal']} nonconc {x['auc_attack_vs_nonconcurrent_normal']}")
    np.savez_compressed(os.path.join(OUT, "scores", f"{protocol}_{name}_s{seed}.npz"), st=st.astype(np.float32))
    return r


def run_ae(kind, S, seed, protocol, label=None):
    m = LSTMAutoencoder(S.n_features) if kind == "LSTM-AE" else DenseAutoencoder(S.X_train.shape[1] * S.n_features)
    t0 = time.time()
    xv_norm = S.X_val if protocol == "F" else S.X_val[S.y_val == 0]
    m, hist = train_ae(m, S.X_train, xv_norm, epochs=60, patience=5, seed=seed, log=log)
    sec = time.time() - t0
    sv, st = reconstruction_error(m, S.X_val), reconstruction_error(m, S.X_test)
    return score(label or kind, seed, sv, st, S, protocol,
                 {"n_params": int(sum(p.numel() for p in m.parameters())), "train_sec": sec, "epochs_run": len(hist)})


def save(name, obj):
    json.dump(obj, open(os.path.join(OUT, name), "w"), indent=1, default=float)


def part_main(seeds):
    for protocol in ["F", "A", "B"]:
        log(f"===== KUBE-IDS0 PROTOCOL {protocol} =====")
        runs, info = [], None
        for seed in seeds:
            S = kids0.build(protocol, T=10, seed=seed)  # F: the 80/20 normal split depends on the seed
            info = S.info
            if seed == seeds[0]:
                log("DATA", json.dumps(S.info))
                np.savez_compressed(os.path.join(OUT, "scores", f"{protocol}_labels.npz"), y=S.y_test, s=S.s_test,
                                    c=S.conc_test, sess=S.sess_test.astype(str), a=S.a_test)
            for kind in ["LSTM-AE", "Dense-AE"]:
                log(f"Training {protocol} {kind} seed {seed}")
                runs.append(run_ae(kind, S, seed, protocol))
            t0 = time.time()
            sv, st = iforest_scores(S.X_train, [S.X_val, S.X_test], seed=seed)
            runs.append(score("IForest", seed, sv, st, S, protocol, {"train_sec": time.time() - t0}))
            if seed == seeds[0]:
                t0 = time.time()
                (sv, st), k = pca_scores(S.X_train, [S.X_val, S.X_test])
                runs.append(score("PCA", 0, sv, st, S, protocol, {"n_components": k, "train_sec": time.time() - t0}))
            del S
        save(f"main_{protocol}.json", {"protocol": protocol, "data": info, "scenarios": kids0.SCEN, "runs": runs})


def part_ablation(seeds):
    runs = []
    for protocol, T, group, label in [("A", 1, "all", "LSTM-AE(T=1)"), ("A", 10, "net", "LSTM-AE(net)"),
                                      ("A", 10, "ctr", "LSTM-AE(ctr)"), ("F", 10, "net", "LSTM-AE(net)")]:
        log(f"===== ABLATION {protocol} T={T} features={group} =====")
        for seed in seeds:
            S = kids0.build(protocol, T=T, group=group, seed=seed)
            r = run_ae("LSTM-AE", S, seed, protocol, label=label)
            r.update({"T": T, "features": group, "data": S.info}); runs.append(r)
            del S
    save("ablation.json", {"runs": runs})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["main", "ablation"], required=True)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    {"main": part_main, "ablation": part_ablation}[a.part](list(range(a.seeds)))
    log("DONE", a.part)
