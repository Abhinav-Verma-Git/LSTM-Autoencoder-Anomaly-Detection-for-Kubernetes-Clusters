"""Aggregate results_v2/*.json -> results_v2/summary.json and results_v2/summary.md."""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import roc_curve

R = "results_v2"
KEYS = ["auc_exploit_vs_capture_background", "auc_capture_background_vs_benign", "auc_exploit_vs_benign_captures", "roc_auc", "pr_auc", "roc_auc_excl_recon", "roc_auc_recon", "precision_attack", "recall_attack",
        "f1_attack", "recall_normal", "fpr", "accuracy", "weighted_f1", "macro_f1", "recall_exploits",
        "recall_recon", "fpr_attack_capture_background", "fpr_benign_captures", "threshold"]


def ms(v):
    v = np.asarray(v, float)
    return [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]


def agg(runs, thr):
    out = {}
    for m in dict.fromkeys(r["model"] for r in runs):
        rs = [r for r in runs if r["model"] == m]
        d = {k: ms([r[thr][k] for r in rs]) for k in KEYS if k in rs[0][thr]}
        d["n_seeds"] = len(rs)
        d["roc_auc_anyflow_label"] = ms([r["test_roc_auc_anyflow_label"] for r in rs])
        for k in ["train_sec", "epochs_run", "inference_ms_per_window"]:
            if k in rs[0]:
                d[k] = ms([r[k] for r in rs])
        for k in ["n_params", "n_components"]:
            if k in rs[0]:
                d[k] = rs[0][k]
        d["per_scenario_n"] = rs[0][thr]["per_scenario_n"]
        d["per_scenario_recall"] = {s: ms([r[thr]["per_scenario_recall"][s] for r in rs]) for s in d["per_scenario_n"]}
        d["per_scenario_auc"] = {s: ms([r[thr]["per_scenario_auc"][s] for r in rs]) for s in d["per_scenario_n"]}
        d["confusion_seed0"] = {k: rs[0][thr][k] for k in ["tn", "fp", "fn", "tp"]}
        out[m] = d
    return out


f = lambda x, p=3: f"{x[0]:.{p}f} ± {x[1]:.{p}f}" if x[1] else f"{x[0]:.{p}f}"


def main():
    S, md = {}, []
    for p in "AB":
        fn = os.path.join(R, f"main_{p}.json")
        if not os.path.exists(fn):
            continue
        J = json.load(open(fn))
        S[p] = {"data": J["data"], "f1": agg(J["runs"], "test_f1_threshold"), "p99": agg(J["runs"], "test_p99_threshold")}
        md.append(f"\n## Protocol {p}\n```\n{json.dumps(J['data'])}\n```")
        md.append("| Model | ROC-AUC | PR-AUC | exploits AUC | recon AUC | any-flow-label AUC |\n|---|---|---|---|---|---|")
        for m, d in S[p]["f1"].items():
            md.append(f"| {m} | {f(d['roc_auc'])} | {f(d['pr_auc'])} | {f(d['roc_auc_excl_recon'])} | {f(d['roc_auc_recon'])} | {f(d['roc_auc_anyflow_label'])} |")
        for thr in ["f1", "p99"]:
            md.append(f"\n{thr}\n| Model | P | R | F1 | FPR | FPR bg-in-attack-caps | FPR benign caps | R exploits | R recon |\n|---|---|---|---|---|---|---|---|---|")
            for m, d in S[p][thr].items():
                md.append(f"| {m} | {f(d['precision_attack'])} | {f(d['recall_attack'])} | {f(d['f1_attack'])} | {f(d['fpr'])} | "
                          f"{f(d['fpr_attack_capture_background'])} | {f(d['fpr_benign_captures'])} | {f(d['recall_exploits'])} | {f(d['recall_recon'])} |")
        d = S[p]["f1"]["LSTM-AE"]
        md.append("\nper scenario (LSTM-AE): " + json.dumps({s: [d['per_scenario_n'][s], round(d['per_scenario_auc'][s][0], 3), round(d['per_scenario_recall'][s][0], 3)] for s in d['per_scenario_n']}))

    for name in ["capture_labels", "ablation", "extra"]:
        fn = os.path.join(R, f"{name}.json")
        if os.path.exists(fn):
            J = json.load(open(fn))
            S[name] = {"f1": agg(J["runs"], "test_f1_threshold"), "p99": agg(J["runs"], "test_p99_threshold")}
            if "data" in J:
                S[name]["data"] = J["data"]
            md.append(f"\n## {name}\n| Config | ROC-AUC | PR-AUC | exploits | recon | F1 | epochs |\n|---|---|---|---|---|---|---|")
            for m, d in S[name]["f1"].items():
                md.append(f"| {m} | {f(d['roc_auc'])} | {f(d['pr_auc'])} | {f(d['roc_auc_excl_recon'])} | {f(d['roc_auc_recon'])} | {f(d['f1_attack'])} | {d.get('epochs_run', [0])[0]:.0f} |")

    # T=10 reference for the ablation = first 3 seeds of main A
    if os.path.exists(os.path.join(R, "main_A.json")):
        J = json.load(open(os.path.join(R, "main_A.json")))
        ref = [dict(r, model="LSTM-AE(T=10)") for r in J["runs"] if r["model"] == "LSTM-AE"][:3]
        S["ablation_T10_ref"] = agg(ref, "test_f1_threshold")["LSTM-AE(T=10)"]
        # original operating point: FPR 0.55, attack prevalence 42705/69384
        prev = 42705 / 69384
        S["matched"] = {"prevalence": prev, "trivial_f1": 2 * prev / (1 + prev)}
        for p in "AB":
            fn = os.path.join(R, f"scores_seed0_{p}.npz")
            if not os.path.exists(fn):
                continue
            z = np.load(fn)
            for m in ["LSTM-AE", "Dense-AE", "IForest", "PCA"]:
                fpr, tpr, _ = roc_curve(z["y_test"], z[f"{m}__test"])
                rec = float(np.interp(0.55, fpr, tpr))
                prc = prev * rec / (prev * rec + (1 - prev) * 0.55)
                S["matched"][f"{p}/{m}"] = {"recall": rec, "precision": prc, "f1": 2 * prc * rec / (prc + rec),
                                            "accuracy": prev * rec + (1 - prev) * 0.45}
            sc = z["LSTM-AE__test"]
            S[f"corr_lstm_dense_{p}"] = float(np.corrcoef(sc, z["Dense-AE__test"])[0, 1])
            # session-shift controls from seed-0 scores (main runs predate the per-seed metric)
            from sklearn.metrics import roc_auc_score
            y, s, c = z["y_test"], z["s_test"], z["c_test"]
            ex, bg, ben = (y == 1) & (s != 2), (y == 0) & (c > 0), (y == 0) & (c == 0)
            auc = lambda neg, pos, e: float(roc_auc_score(np.r_[np.zeros(neg.sum()), np.ones(pos.sum())], np.r_[e[neg], e[pos]]))
            S[f"controls_seed0_{p}"] = {m: {"exploit_vs_benign_captures": auc(ben, ex, z[f"{m}__test"]),
                                            "exploit_vs_capture_background": auc(bg, ex, z[f"{m}__test"]),
                                            "capture_background_vs_benign": auc(ben, bg, z[f"{m}__test"]),
                                            "n_exploit": int(ex.sum()), "n_bg": int(bg.sum()), "n_benign": int(ben.sum())}
                                        for m in ["LSTM-AE", "Dense-AE", "IForest", "PCA"]}
            md.append(f"\ncontrols seed0 {p}: " + json.dumps(S[f"controls_seed0_{p}"]))
        md.append("\n## matched\n" + json.dumps(S["matched"], indent=1))

    json.dump(S, open(os.path.join(R, "summary.json"), "w"), indent=1)
    open(os.path.join(R, "summary.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
