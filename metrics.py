"""Threshold selection (Eq. 18), decision rule (Eq. 19) and evaluation metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)


def f1_optimal_threshold(scores: np.ndarray, y: np.ndarray, n_candidates: int = 2000) -> float:
    """theta* = argmax_theta F1(y, I(eps >= theta)) over candidate thresholds (Eq. 18)."""
    cand = np.unique(np.quantile(scores, np.linspace(0, 1, n_candidates)))
    best_t, best_f = cand[0], -1.0
    for t in cand:
        f = f1_score(y, (scores >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t)


def percentile_threshold(normal_scores: np.ndarray, q: float = 99.0) -> float:
    """Label-free threshold: q-th percentile of reconstruction error on NORMAL validation windows."""
    return float(np.percentile(normal_scores, q))


def evaluate(scores: np.ndarray, y: np.ndarray, theta: float, scen: np.ndarray | None = None,
             cap: np.ndarray | None = None) -> dict:
    yhat = (scores >= theta).astype(int)  # Eq. 19
    p, r, f, sup = precision_recall_fscore_support(y, yhat, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out = {
        "threshold": theta,
        "accuracy": accuracy_score(y, yhat),
        "precision_attack": p[1], "recall_attack": r[1], "f1_attack": f[1],
        "precision_normal": p[0], "recall_normal": r[0], "f1_normal": f[0],
        "macro_f1": f1_score(y, yhat, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, yhat, average="weighted", zero_division=0),
        "fpr": fp / max(fp + tn, 1),
        "roc_auc": roc_auc_score(y, scores),
        "pr_auc": average_precision_score(y, scores),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "support_normal": int(sup[0]), "support_attack": int(sup[1]),
    }
    if scen is not None:
        out["per_scenario_recall"] = {int(s): float(yhat[scen == s].mean()) for s in np.unique(scen) if s > 0}
        out["per_scenario_n"] = {int(s): int((scen == s).sum()) for s in np.unique(scen) if s > 0}
        nm = scen == 0
        out["per_scenario_auc"] = {
            int(s): float(roc_auc_score(np.r_[np.zeros(nm.sum()), np.ones((scen == s).sum())],
                                        np.r_[scores[nm], scores[scen == s]]))
            for s in np.unique(scen) if s > 0}
        ex = (scen != 2)  # all windows except Node-RED reconnaissance
        out["roc_auc_excl_recon"] = float(roc_auc_score(y[ex], scores[ex]))
        rc = (scen == 0) | (scen == 2)  # reconnaissance only
        out["roc_auc_recon"] = float(roc_auc_score(y[rc], scores[rc]))
        out["recall_exploits"] = float(yhat[(y == 1) & (scen != 2)].mean())
        out["recall_recon"] = float(yhat[scen == 2].mean())
    if cap is not None:
        bg = (y == 0) & (cap > 0)  # benign (relabelled) flows recorded inside attack captures
        if bg.any():
            out["fpr_attack_capture_background"] = float(yhat[bg].mean())
            out["fpr_benign_captures"] = float(yhat[(y == 0) & (cap == 0)].mean())
            if scen is not None:
                ex = (y == 1) & (scen != 2)
                ben = (y == 0) & (cap == 0)
                auc = lambda neg, pos: float(roc_auc_score(np.r_[np.zeros(neg.sum()), np.ones(pos.sum())],
                                                           np.r_[scores[neg], scores[pos]]))
                # Session-shift controls: exploits vs benign flows from the SAME attack captures, and
                # attack-capture background vs benign-capture flows (no attacks on either side).
                out["auc_exploit_vs_capture_background"] = auc(bg, ex)
                out["auc_capture_background_vs_benign"] = auc(ben, bg)
                out["auc_exploit_vs_benign_captures"] = auc(ben, ex)
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}
