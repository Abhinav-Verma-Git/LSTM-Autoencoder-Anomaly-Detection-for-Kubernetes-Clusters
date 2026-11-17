"""Aggregate results_kids0/*.json -> results_kids0/summary.json (+ summary.md)."""
from __future__ import annotations

import json
import os

import numpy as np

R = "results_kids0"
KEYS = ["roc_auc", "pr_auc", "precision_attack", "recall_attack", "f1_attack", "recall_normal", "fpr", "accuracy",
        "weighted_f1", "macro_f1", "threshold"]
XKEYS = ["auc_attack_vs_concurrent_normal", "auc_attack_vs_nonconcurrent_normal", "auc_concurrent_vs_nonconcurrent_normal",
         "roc_auc_anyflow_label", "fpr_concurrent_f1", "fpr_nonconcurrent_f1", "fpr_concurrent_p99", "fpr_nonconcurrent_p99"]


def ms(v):
    v = np.asarray([x for x in v if x is not None], float)
    if len(v) == 0:
        return None
    return [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]


def agg(runs):
    out = {}
    for m in dict.fromkeys(r["model"] for r in runs):
        rs = [r for r in runs if r["model"] == m]
        d = {"n_seeds": len(rs)}
        for thr in ["test_f1_threshold", "test_p99_threshold"]:
            d[thr] = {k: ms([r[thr][k] for r in rs]) for k in KEYS}
            d[thr]["per_scenario_recall"] = {s: ms([r[thr]["per_scenario_recall"][s] for r in rs]) for s in rs[0][thr]["per_scenario_recall"]}
        d["extra"] = {k: ms([r["extra"][k] for r in rs]) for k in XKEYS}
        d["per_scenario_auc"] = {s: ms([r["extra"]["per_scenario_auc"][s] for r in rs]) for s in rs[0]["extra"]["per_scenario_auc"]}
        d["per_scenario_auc_vs_concurrent"] = {s: ms([r["extra"]["per_scenario_auc_vs_concurrent"][s] for r in rs])
                                               for s in rs[0]["extra"]["per_scenario_auc_vs_concurrent"]}
        for k in ["train_sec", "epochs_run"]:
            if k in rs[0]:
                d[k] = ms([r[k] for r in rs])
        if "n_params" in rs[0]:
            d["n_params"] = rs[0]["n_params"]
        d["per_seed_fpr_f1"] = [r["test_f1_threshold"]["fpr"] for r in rs]
        out[m] = d
    return out


def main():
    S, md = {}, []
    for p in "FAB":
        fn = os.path.join(R, f"main_{p}.json")
        if not os.path.exists(fn):
            continue
        J = json.load(open(fn))
        S[p] = {"data": J["data"], "models": agg(J["runs"])}
        y = J["data"]["test"]
        prev = y["attack"] / (y["attack"] + y["normal"])
        S[p]["prevalence"] = prev
        S[p]["trivial"] = {"precision": prev, "recall": 1.0, "f1": 2 * prev / (1 + prev), "accuracy": prev,
                           "recall_normal": 0.0, "weighted_f1": None}
        md.append(f"\n## Protocol {p}  (attack share {prev:.3f}, flag-all F1 {2 * prev / (1 + prev):.3f})\n```\n{json.dumps(J['data'])}\n```")
        md.append("| Model | ROC | PR | P | R | F1 | Rn | Acc | wF1 | thr | conc | nonconc | conc-vs-nonconc normal |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        f = lambda v: "-" if v is None else (f"{v[0]:.3f}±{v[1]:.3f}" if v[1] else f"{v[0]:.3f}")
        for m, d in S[p]["models"].items():
            t, x = d["test_f1_threshold"], d["extra"]
            md.append(f"| {m} | {f(t['roc_auc'])} | {f(t['pr_auc'])} | {f(t['precision_attack'])} | {f(t['recall_attack'])} | {f(t['f1_attack'])} | "
                      f"{f(t['recall_normal'])} | {f(t['accuracy'])} | {f(t['weighted_f1'])} | {f(t['threshold'])} | {f(x['auc_attack_vs_concurrent_normal'])} | "
                      f"{f(x['auc_attack_vs_nonconcurrent_normal'])} | {f(x['auc_concurrent_vs_nonconcurrent_normal'])} |")
        md.append("p99: " + json.dumps({m: {k: d["test_p99_threshold"][k] for k in ["precision_attack", "recall_attack", "fpr"]} for m, d in S[p]["models"].items()}))
        md.append("per-scenario AUC LSTM-AE: " + json.dumps(S[p]["models"]["LSTM-AE"]["per_scenario_auc"]))
        md.append("per-scenario AUC vs concurrent LSTM-AE: " + json.dumps(S[p]["models"]["LSTM-AE"]["per_scenario_auc_vs_concurrent"]))
    fn = os.path.join(R, "ablation.json")
    if os.path.exists(fn):
        J = json.load(open(fn))
        groups = {}
        for r in J["runs"]:
            groups.setdefault(f"{r['protocol']}|{r['model']}", []).append(r)
        S["ablation"] = {k: agg(v)[v[0]["model"]] for k, v in groups.items()}
        md.append("\n## ablation")
        for k, d in S["ablation"].items():
            t, x = d["test_f1_threshold"], d["extra"]
            md.append(f"{k}: ROC {t['roc_auc']} PR {t['pr_auc']} F1 {t['f1_attack']} conc {x['auc_attack_vs_concurrent_normal']} "
                      f"nonconc {x['auc_attack_vs_nonconcurrent_normal']} per-scen {json.dumps(d['per_scenario_auc'])}")
    json.dump(S, open(os.path.join(R, "summary.json"), "w"), indent=1)
    open(os.path.join(R, "summary.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
