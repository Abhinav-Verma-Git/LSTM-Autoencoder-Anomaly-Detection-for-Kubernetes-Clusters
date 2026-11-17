"""Live demo for the presentation: reproduces the key Kube-IDS0 results in about 5 minutes on a laptop CPU.

  python demo.py              # full live demo (about 5 minutes)
  python demo.py --fast       # shorter training (about 2 minutes), numbers slightly less exact
  python demo.py --saved      # no training: print the saved results of the full experiments instantly
  python demo.py --relabel    # Sever-Dogan label audit (needs the data_k8s folder; takes seconds)

What it shows, live:
  1. Kube-IDS0 has exactly the original article's 42,705 attack samples.
  2. Following the article's procedure, our LSTM-AE reproduces its precision / recall / F1.
  3. A "flag everything" detector and a plain dense autoencoder score about as well or better.
  4. Scoring single flows (T=1) beats the article's 10-flow windows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PAPER = {"precision": 0.73, "recall": 0.95, "f1": 0.83, "normal_recall": 0.45, "accuracy": 0.76}


def line(ch="="):
    print(ch * 78)


def step(n, total, title):
    print()
    line()
    print(f"[{n}/{total}] {title}")
    line()


def metrics_row(name, d):
    return (f"{name:<34}{d['precision']:>9.3f}{d['recall']:>8.3f}{d['f1']:>8.3f}"
            f"{d['normal_recall']:>9.3f}{d['accuracy']:>8.3f}{d.get('roc_auc', float('nan')):>9.3f}")


def header():
    print(f"{'Detector':<34}{'Prec.':>9}{'Recall':>8}{'F1':>8}{'NormRec':>9}{'Acc.':>8}{'ROC-AUC':>9}")
    print("-" * 85)


def as_row(ev):
    return {"precision": ev["precision_attack"], "recall": ev["recall_attack"], "f1": ev["f1_attack"],
            "normal_recall": ev["recall_normal"], "accuracy": ev["accuracy"], "roc_auc": ev["roc_auc"]}


def show_saved():
    p = os.path.join(HERE, "results_kids0", "summary.json")
    S = json.load(open(p))
    F = S["F"]["models"]
    step(1, 3, "Saved results: direct replication on Kube-IDS0, article's procedure (5 seeds, mean)")
    header()
    print(metrics_row("Original article (reported)", {**PAPER, "roc_auc": float("nan")}))
    for m in ["LSTM-AE", "Dense-AE", "PCA", "IForest"]:
        t = F[m]["test_f1_threshold"]
        print(metrics_row(f"{m} (ours)", {"precision": t["precision_attack"][0], "recall": t["recall_attack"][0],
                                          "f1": t["f1_attack"][0], "normal_recall": t["recall_normal"][0],
                                          "accuracy": t["accuracy"][0], "roc_auc": t["roc_auc"][0]}))
    pr = S["F"]["prevalence"]
    print(metrics_row("Flag every flow as attack", {"precision": pr, "recall": 1.0, "f1": 2 * pr / (1 + pr),
                                                     "normal_recall": 0.0, "accuracy": pr, "roc_auc": 0.5}))
    step(2, 3, "Saved results: leakage-free Protocol A (ROC-AUC, 5 seeds)")
    for m in ["LSTM-AE", "Dense-AE", "PCA", "IForest"]:
        print(f"  {m:<10} ROC-AUC {S['A']['models'][m]['test_f1_threshold']['roc_auc'][0]:.3f}")
    print(f"  LSTM-AE with single flows (T=1): ROC-AUC {S['ablation']['A|LSTM-AE(T=1)']['test_f1_threshold']['roc_auc'][0]:.3f}")
    V = json.load(open(os.path.join(HERE, "results_v2", "summary.json")))
    step(3, 3, "Saved results: cross-dataset (Sever-Dogan, relabelled), LSTM-AE ROC-AUC")
    print(f"  Protocol A: {V['A']['f1']['LSTM-AE']['roc_auc'][0]:.3f}   Protocol B: {V['B']['f1']['LSTM-AE']['roc_auc'][0]:.3f}"
          "   (0.5 = random guessing)")


def show_relabel():
    from relabel import flow_labels
    from data import SCENARIO_NAMES, load_captures
    root = os.path.join(HERE, "data_k8s")
    if not os.path.isdir(root):
        sys.exit("data_k8s folder not found. Download https://github.com/yigitsever/kubernetes-dataset into it.")
    step(1, 1, "Sever-Dogan label audit: how many 'attack' flows are really attacks?")
    print(f"{'Scenario':<44}{'Flows':>7}{'Attack':>8}{'Benign':>8}")
    print("-" * 67)
    tot = att = 0
    for c in sorted(load_captures(root), key=lambda c: c.label):
        if c.label == 0:
            continue
        lab, _ = flow_labels(c.frame, c.label)
        n, a = len(lab), int((lab > 0).sum())
        tot += n; att += a
        print(f"{SCENARIO_NAMES[c.label]:<44}{n:>7}{a:>8}{n - a:>8}")
    print("-" * 67)
    print(f"{'TOTAL':<44}{tot:>7}{att:>8}{tot - att:>8}")
    print(f"\n=> {tot - att} of {tot} attack-labelled flows ({100 * (tot - att) / tot:.1f}%) are normal control-plane traffic.")


def live(fast: bool):
    import numpy as np
    import torch
    import kids0
    from metrics import evaluate, f1_optimal_threshold
    from models import DenseAutoencoder, LSTMAutoencoder, reconstruction_error, train_ae

    torch.set_num_threads(os.cpu_count() or 2)
    epochs = 15 if fast else 60
    t_start = time.time()
    total = 5

    step(1, total, "Load Kube-IDS0 (the original article's dataset)")
    t0 = time.time()
    df = kids0.load()
    n_att = int((df.scen > 0).sum())
    print(f"Flows: {len(df):,}   attacks: {n_att:,}   normal: {len(df) - n_att:,}   features: {len(kids0.feature_cols(df))}")
    print(f"Original article's evaluation set: 42,705 attacks  ->  {'EXACT MATCH' if n_att == 42705 else 'DIFFERENT'}")
    print(f"({time.time() - t0:.0f} s; the first run also aligns timestamps and caches the result)")

    step(2, total, f"Train the LSTM autoencoder on NORMAL traffic only (article's procedure, up to {epochs} epochs)")
    S = kids0.build("F", T=10, seed=0)
    print(f"Training windows: {len(S.X_train):,} (normal only)   evaluation windows: {len(S.X_test):,}")
    t0 = time.time()
    m, hist = train_ae(LSTMAutoencoder(S.n_features), S.X_train, S.X_val, epochs=epochs, seed=0)
    print(f"Trained in {time.time() - t0:.0f} s ({len(hist)} epochs)")

    step(3, total, "Score every flow and compare with the article")
    st = reconstruction_error(m, S.X_test)
    ev_lstm = evaluate(st, S.y_test, f1_optimal_threshold(st, S.y_test))
    header()
    print(metrics_row("Original article (reported)", {**PAPER, "roc_auc": float("nan")}))
    print(metrics_row("Our LSTM-AE (live, seed 0)", as_row(ev_lstm)))
    print("\n=> Same procedure, same data: we reproduce the article's numbers.")

    step(4, total, "Is the LSTM really needed? Two simple comparisons")
    prev = float(S.y_test.mean())
    flag_all = {"precision": prev, "recall": 1.0, "f1": 2 * prev / (1 + prev), "normal_recall": 0.0,
                "accuracy": prev, "roc_auc": 0.5}
    t0 = time.time()
    md, _ = train_ae(DenseAutoencoder(S.X_train.shape[1] * S.n_features), S.X_train, S.X_val, epochs=epochs, seed=0,
                     log=lambda *a: None)
    sd = reconstruction_error(md, S.X_test)
    ev_dense = evaluate(sd, S.y_test, f1_optimal_threshold(sd, S.y_test))
    print(f"(dense autoencoder trained in {time.time() - t0:.0f} s)\n")
    header()
    print(metrics_row("Original article (reported)", {**PAPER, "roc_auc": float("nan")}))
    print(metrics_row("Our LSTM-AE (live)", as_row(ev_lstm)))
    print(metrics_row("Dense autoencoder, no LSTM (live)", as_row(ev_dense)))
    print(metrics_row("Flag every flow as attack", flag_all))
    print(f"\n=> {100 * prev:.1f}% of flows are attacks, so flagging everything already gives F1 {flag_all['f1']:.3f}.")
    print(f"=> The article's threshold flags {100 * (1 - ev_lstm['recall_normal']):.0f}% of normal traffic as attacks.")
    del S

    step(5, total, "Leakage-free test (Protocol A): 10-flow windows vs single flows")
    res = {}
    for T in [10, 1]:
        SA = kids0.build("A", T=T, seed=0)
        t0 = time.time()
        ma, _ = train_ae(LSTMAutoencoder(SA.n_features), SA.X_train, SA.X_val[SA.y_val == 0], epochs=epochs, seed=0,
                         log=lambda *a: None)
        sv, sa = reconstruction_error(ma, SA.X_val), reconstruction_error(ma, SA.X_test)
        res[T] = evaluate(sa, SA.y_test, f1_optimal_threshold(sv, SA.y_val))
        print(f"  LSTM-AE, window T={T:<2}: ROC-AUC {res[T]['roc_auc']:.3f}   F1 {res[T]['f1_attack']:.3f}   "
              f"({time.time() - t0:.0f} s)")
        del SA
    print("\n=> Scoring single flows works better than the article's 10-flow windows.")

    print()
    line()
    print(f"Demo finished in {(time.time() - t_start) / 60:.1f} minutes.")
    print("Numbers can differ slightly from the paper's tables: those are means of 5 seeds, and results vary a")
    print("little between computers. Run 'python demo.py --saved' to show the full saved results.")
    line()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="shorter training (about 2 minutes)")
    ap.add_argument("--saved", action="store_true", help="print saved results, no training")
    ap.add_argument("--relabel", action="store_true", help="Sever-Dogan label audit (needs data_k8s)")
    a = ap.parse_args()
    if a.saved:
        show_saved()
    elif a.relabel:
        show_relabel()
    else:
        live(a.fast)
