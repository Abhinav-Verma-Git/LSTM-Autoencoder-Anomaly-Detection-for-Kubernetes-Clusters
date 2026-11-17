"""Generate LaTeX tables (tables/*.tex) and number macros (numbers.tex) from results_v2/summary.json.
Every number in the paper's results comes from this script, so the text cannot drift from the data."""
import json
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "tex"
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
S = json.load(open("results_v2/summary.json"))
RC = json.load(open("results_relabel/relabel_counts.json"))
MODELS = [("LSTM-AE", "LSTM-AE"), ("Dense-AE", "Dense-AE"), ("IForest", "IForest"), ("PCA", "PCA")]
macros = {}


def num(x, p=3):
    return f"{x:.{p}f}"


def pm(v, p=3, bold=False):
    s = num(v[0], p) if v[1] == 0 else f"\\pmv{{{num(v[0], p)}}}{{{num(v[1], p)}}}"
    return f"\\textbf{{{s}}}" if bold else s


def intc(n):
    return f"{n:,}".replace(",", "{,}")


def M(name, val):
    macros[name] = val


def best(vals, hi=True):
    arr = [v[0] for v in vals]
    return int(np.argmax(arr) if hi else np.argmin(arr))


# ---------------- relabelling counts ----------------
tot = sum(r["flows"] for r in RC); att = sum(r["attack"] for r in RC); bg = sum(r["relabelled_benign"] for r in RC)
recon = next(r["attack"] for r in RC if r["scenario"] == 2)
M("nAttackCapFlows", intc(tot)); M("nTrueAttack", intc(att)); M("nRelabelled", intc(bg))
M("pctRelabelled", f"{100 * bg / tot:.1f}"); M("nRecon", intc(recon)); M("pctRecon", f"{100 * recon / att:.1f}")
M("nExploitFlows", intc(att - recon))
M("nRone", intc(sum(r["attack_R1"] for r in RC))); M("nRtwo", intc(sum(r["attack_R2"] for r in RC)))

# ---------------- split counts table ----------------
rows = []
for p in "AB":
    d = S[p]["data"]
    for sp in ["val", "test"]:
        c = d[sp]; abs_ = c["attack_by_scenario"]
        ex = sum(v for k, v in abs_.items() if k != "2")
        rows.append(f"{p} & {'Validation' if sp == 'val' else 'Test'} & {intc(c['normal'])} & {intc(c['normal_from_attack_captures'])} & "
                    f"{intc(abs_.get('2', 0))} & {intc(ex)} \\\\")
M("nTrain", intc(S["A"]["data"]["n_train_windows"]))
M("nTestExploitA", intc(sum(v for k, v in S["A"]["data"]["test"]["attack_by_scenario"].items() if k != "2")))
M("nTestExploitB", intc(sum(v for k, v in S["B"]["data"]["test"]["attack_by_scenario"].items() if k != "2")))
M("pctAttackTestA", f"{100 * S['A']['data']['test']['attack'] / (S['A']['data']['test']['attack'] + S['A']['data']['test']['normal']):.1f}")
M("pctAttackTestB", f"{100 * S['B']['data']['test']['attack'] / (S['B']['data']['test']['attack'] + S['B']['data']['test']['normal']):.1f}")
open(os.path.join(OUT, "tables", "splits.tex"), "w").write(
    "\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\n & & \\multicolumn{2}{c}{Benign flows} & \\multicolumn{2}{c}{Attack flows} \\\\\n"
    "\\cmidrule(lr){3-4}\\cmidrule(l){5-6}\nProt. & Split & Total & from attack caps. & Recon. & Exploits \\\\\n\\midrule\n"
    f"-- & Training & {intc(S['A']['data']['n_train_windows'])} & 0 & 0 & 0 \\\\\n\\midrule\n" + "\n".join(rows) +
    "\n\\bottomrule\n\\end{tabular}\n")

# ---------------- ranking table ----------------
lines = []
for p in "AB":
    F = S[p]["f1"]
    cols = ["roc_auc", "pr_auc", "roc_auc_excl_recon", "roc_auc_recon"]
    bests = {k: best([F[m][k] for m, _ in MODELS]) for k in cols}
    for i, (m, lab) in enumerate(MODELS):
        cells = " & ".join(pm(F[m][k], bold=(bests[k] == i)) for k in cols)
        pre = f"\\multirow{{4}}{{*}}{{{p}}}" if i == 0 else ""
        lines.append(f"{pre} & {lab} & {cells} \\\\")
    if p == "A":
        lines.append("\\midrule")
open(os.path.join(OUT, "tables", "ranking.tex"), "w").write(
    "\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n & & \\multicolumn{2}{c}{All attack flows} & Exploits & Recon. \\\\\n"
    "\\cmidrule(lr){3-4}\\cmidrule(lr){5-5}\\cmidrule(l){6-6}\n & Model & ROC-AUC & PR-AUC & ROC-AUC & ROC-AUC \\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
for p in "AB":
    L = S[p]["f1"]["LSTM-AE"]
    M(f"lstmAuc{p}", pm(L["roc_auc"])); M(f"lstmAucMean{p}", num(L["roc_auc"][0]))
    M(f"lstmPr{p}", num(L["pr_auc"][0])); M(f"lstmExpl{p}", num(L["roc_auc_excl_recon"][0]))
    M(f"lstmRecon{p}", num(L["roc_auc_recon"][0]))
    M(f"lstmAnyFlow{p}", num(L["roc_auc_anyflow_label"][0]))
    M(f"corrLD{p}", num(S[f"corr_lstm_dense_{p}"]))
    ex = [S[p]["f1"][m]["roc_auc_excl_recon"][0] for m, _ in MODELS]
    M(f"explMin{p}", num(min(ex))); M(f"explMax{p}", num(max(ex)))
    rc = [S[p]["f1"][m]["roc_auc_recon"][0] for m, _ in MODELS]
    M(f"reconMin{p}", num(min(rc))); M(f"reconMax{p}", num(max(rc)))
    pr = S[p]["data"]["test"]["attack"] / (S[p]["data"]["test"]["attack"] + S[p]["data"]["test"]["normal"])
    M(f"prChance{p}", num(pr))

# ---------------- session-shift control table (seed 0) ----------------
lines = []
for p in "AB":
    C = S[f"controls_seed0_{p}"]
    for i, (m, lab) in enumerate(MODELS):
        c = C[m]
        pre = f"\\multirow{{4}}{{*}}{{{p}}}" if i == 0 else ""
        lines.append(f"{pre} & {lab} & {num(c['exploit_vs_benign_captures'])} & {num(c['exploit_vs_capture_background'])} & "
                     f"{num(c['capture_background_vs_benign'])} \\\\")
    if p == "A":
        lines.append("\\midrule")
    M(f"ctrlExplBen{p}", num(C["LSTM-AE"]["exploit_vs_benign_captures"]))
    M(f"ctrlExplBg{p}", num(C["LSTM-AE"]["exploit_vs_capture_background"]))
    M(f"ctrlBgBen{p}", num(C["LSTM-AE"]["capture_background_vs_benign"]))
    M(f"ctrlNexpl{p}", intc(C["LSTM-AE"]["n_exploit"])); M(f"ctrlNbg{p}", intc(C["LSTM-AE"]["n_bg"]))
    vals = [C[m]["exploit_vs_capture_background"] for m, _ in MODELS]
    M(f"ctrlExplBgMin{p}", num(min(vals))); M(f"ctrlExplBgMax{p}", num(max(vals)))
    vals = [C[m]["capture_background_vs_benign"] for m, _ in MODELS]
    M(f"ctrlBgBenMin{p}", num(min(vals))); M(f"ctrlBgBenMax{p}", num(max(vals)))
open(os.path.join(OUT, "tables", "controls.tex"), "w").write(
    "\\begin{tabular}{@{}llccc@{}}\n\\toprule\n & & (a) Exploits vs & (b) Exploits vs & (c) Attack-capture \\\\\n"
    " & Model & benign captures & same-capture benign & benign vs benign caps. \\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------- threshold table ----------------
lines = []
for thr, tlab in [("f1", "$\\theta^{*}$ (F1-optimal)"), ("p99", "$\\theta_{99}$ (label-free)")]:
    for i, (m, lab) in enumerate(MODELS):
        cells = []
        for p in "AB":
            d = S[p][thr][m]
            cells += [pm(d["precision_attack"]), pm(d["recall_attack"]), pm(d["f1_attack"]), pm(d["fpr"])]
        pre = f"\\multirow{{4}}{{*}}{{{tlab}}}" if i == 0 else ""
        lines.append(f"{pre} & {lab} & " + " & ".join(cells) + " \\\\")
    if thr == "f1":
        lines.append("\\midrule")
open(os.path.join(OUT, "tables", "threshold.tex"), "w").write(
    "\\begin{tabular}{@{}llcccccccc@{}}\n\\toprule\n & & \\multicolumn{4}{c}{Protocol A} & \\multicolumn{4}{c}{Protocol B} \\\\\n"
    "\\cmidrule(lr){3-6}\\cmidrule(l){7-10}\nThreshold & Model & Precision & Recall & F1 & FPR & Precision & Recall & F1 & FPR \\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
for p in "AB":
    for thr in ["f1", "p99"]:
        d = S[p][thr]["LSTM-AE"]; T = "F" if thr == "f1" else "P"
        for k, short in [("precision_attack", "Prec"), ("recall_attack", "Rec"), ("f1_attack", "Fone"), ("fpr", "Fpr"),
                         ("recall_exploits", "RecExpl"), ("recall_recon", "RecRecon"),
                         ("fpr_attack_capture_background", "FprBg"), ("fpr_benign_captures", "FprBen")]:
            M(f"lstm{T}{short}{p}", num(d[k][0]))
            M(f"lstm{T}{short}Sd{p}", num(d[k][1]))
    M(f"denseFFpr{p}", num(S[p]["f1"]["Dense-AE"]["fpr"][0])); M(f"ifFFpr{p}", num(S[p]["f1"]["IForest"]["fpr"][0]))
    M(f"denseFRec{p}", num(S[p]["f1"]["Dense-AE"]["recall_attack"][0])); M(f"ifFRec{p}", num(S[p]["f1"]["IForest"]["recall_attack"][0]))
    pp = [S[p]["p99"][m]["precision_attack"][0] for m, _ in MODELS if m != "LSTM-AE"]
    M(f"basePnnPrecMax{p}", num(max(pp)))

# per-seed F1-threshold collapse for LSTM-AE
for p in "AB":
    J = json.load(open(f"results_v2/main_{p}.json"))
    fprs = [r["test_f1_threshold"]["fpr"] for r in J["runs"] if r["model"] == "LSTM-AE"]
    M(f"lstmSeedFprs{p}", ", ".join(num(x, 2) for x in fprs))
    M(f"lstmSeedCollapse{p}", str(sum(x > 0.5 for x in fprs)))
    for m, key in [("Dense-AE", "dense"), ("IForest", "if")]:
        fr = [r["test_f1_threshold"]["fpr"] for r in J["runs"] if r["model"] == m]
        M(f"{key}SeedCollapse{p}", str(sum(x > 0.5 for x in fr)))
M("trivialFoneValA", num(2 * (S["A"]["data"]["val"]["attack"] / (S["A"]["data"]["val"]["attack"] + S["A"]["data"]["val"]["normal"])) /
                       (1 + S["A"]["data"]["val"]["attack"] / (S["A"]["data"]["val"]["attack"] + S["A"]["data"]["val"]["normal"]))))

# ---------------- capture labels vs flow labels ----------------
if "capture_labels" in S:
    C = S["capture_labels"]["f1"]["LSTM-AE(capture-labels)"]; R = S["ablation_T10_ref"]
    cd = S["capture_labels"]["data"]["test"]
    pct_cp = 100 * (cd["attack"] - S["A"]["data"]["test"]["attack"]) / cd["attack"]
    M("capPctControlPlane", f"{pct_cp:.1f}")
    rows = [("Capture-level (original)", C, intc(cd["attack"])), ("Flow-level (ours)", R, intc(S["A"]["data"]["test"]["attack"]))]
    body = "\n".join(f"{n} & {a} & {pm(d['roc_auc'])} & {pm(d['pr_auc'])} & {pm(d['f1_attack'])} & {pm(d['fpr'])} \\\\" for n, d, a in rows)
    open(os.path.join(OUT, "tables", "labels.tex"), "w").write(
        "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nLabels & Attack flows & ROC-AUC & PR-AUC & F1 & FPR \\\\\n\\midrule\n" + body +
        "\n\\bottomrule\n\\end{tabular}\n")
    M("capAuc", num(C["roc_auc"][0])); M("capPr", num(C["pr_auc"][0])); M("capFone", num(C["f1_attack"][0]))
    M("flowAucRef", num(R["roc_auc"][0])); M("flowPrRef", num(R["pr_auc"][0])); M("flowFoneRef", num(R["f1_attack"][0]))
    M("capPnnPrec", num(S["capture_labels"]["p99"]["LSTM-AE(capture-labels)"]["precision_attack"][0]))
    M("capRecall", num(C["recall_attack"][0])); M("capExpl", num(C["roc_auc_excl_recon"][0]))
    M("capPrChance", num(cd["attack"] / (cd["attack"] + cd["normal"])))

# ---------------- ablation ----------------
if "ablation" in S and "extra" in S:
    A = S["ablation"]["f1"]; E = S["extra"]["f1"]; R = S["ablation_T10_ref"]
    cfgs = [("$T=1$ (no temporal context)", A["LSTM-AE(T=1)"]), ("$T=5$", A["LSTM-AE(T=5)"]), ("$T=10$ (original setting)", R),
            ("$T=20$", A["LSTM-AE(T=20)"]), ("$T=10$ + log scaling", A["LSTM-AE(log1p)"]),
            ("$T=10$, $\\le$200 epochs", E["LSTM-AE(200ep,mean)"]), ("\\quad + max-step score", E["LSTM-AE(200ep,max)"]),
            ("\\quad + last-step score", E["LSTM-AE(200ep,last)"])]
    keys = ["roc_auc", "roc_auc_excl_recon", "roc_auc_recon", "auc_exploit_vs_capture_background"]
    bests = {k: best([d[k] if k in d else [-1.0, 0.0] for _, d in cfgs]) for k in keys}
    body = []
    for i, (n, d) in enumerate(cfgs):
        cells = " & ".join(pm(d[k], bold=(bests[k] == i)) if k in d else "--" for k in keys)
        body.append(f"{n} & {cells} \\\\")
    open(os.path.join(OUT, "tables", "ablation.tex"), "w").write(
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n & All & Exploits vs & Recon. vs & Exploits vs same- \\\\\n"
        "Variant & attacks & benign & benign & capture benign \\\\\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")
    for T in [1, 5, 20]:
        d = A[f"LSTM-AE(T={T})"]; t = {1: "One", 5: "Five", 20: "Twenty"}[T]
        M(f"ablAll{t}", num(d["roc_auc"][0])); M(f"ablExpl{t}", num(d["roc_auc_excl_recon"][0])); M(f"ablRecon{t}", num(d["roc_auc_recon"][0]))
        M(f"ablCtrl{t}", num(d["auc_exploit_vs_capture_background"][0]))
    M("ablAllTen", num(R["roc_auc"][0])); M("ablExplTen", num(R["roc_auc_excl_recon"][0])); M("ablReconTen", num(R["roc_auc_recon"][0]))
    d = A["LSTM-AE(log1p)"]; M("ablLogExpl", num(d["roc_auc_excl_recon"][0])); M("ablLogRecon", num(d["roc_auc_recon"][0])); M("ablLogAll", num(d["roc_auc"][0]))
    M("ablLogCtrl", num(d["auc_exploit_vs_capture_background"][0]))
    e = E["LSTM-AE(200ep,mean)"]; M("extraAll", num(e["roc_auc"][0])); M("extraEpMin", f"{min(r for r in [e['epochs_run'][0]]):.0f}")
    J = json.load(open("results_v2/extra.json"))
    eps = sorted({r["epochs_run"] for r in J["runs"]}); M("extraEpLo", str(eps[0])); M("extraEpHi", str(eps[-1]))
    M("extraMaxAll", num(E["LSTM-AE(200ep,max)"]["roc_auc"][0])); M("extraLastAll", num(E["LSTM-AE(200ep,last)"]["roc_auc"][0]))
    M("extraCtrl", num(e["auc_exploit_vs_capture_background"][0]))
    M("extraLastCtrl", num(E["LSTM-AE(200ep,last)"]["auc_exploit_vs_capture_background"][0]))
    M("extraMaxCtrl", num(E["LSTM-AE(200ep,max)"]["auc_exploit_vs_capture_background"][0]))

# ---------------- matched operating point ----------------
mt = S["matched"]
M("trivialF", num(mt["trivial_f1"])); M("prevOrig", f"{100 * mt['prevalence']:.1f}"); M("trivialP", num(mt["prevalence"]))
rows = ["LSTM-AE of~\\cite{svyrydov2026} (reported) & 0.95 & 0.73 & 0.83 & 0.76 \\\\",
        f"Flag every window as attack & 1.000 & {num(mt['prevalence'])} & {num(mt['trivial_f1'])} & {num(mt['prevalence'])} \\\\"]
for p in "AB":
    d = mt[f"{p}/LSTM-AE"]
    rows.append(f"Our LSTM-AE, Protocol {p} & {num(d['recall'])} & {num(d['precision'])} & {num(d['f1'])} & {num(d['accuracy'])} \\\\")
    M(f"matchRec{p}", num(d["recall"])); M(f"matchFone{p}", num(d["f1"]))
open(os.path.join(OUT, "tables", "matched.tex"), "w").write(
    "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nDetector & Recall & Precision & F1 & Accuracy \\\\\n\\midrule\n" + "\n".join(rows) +
    "\n\\bottomrule\n\\end{tabular}\n")

# ---------------- alert volume (benign captures: 60,471 flows over their recorded duration) ----------------
import glob
import pandas as pd
n_fl, secs = 0, 0.0
for fn in glob.glob("data_k8s/benign/*.csv"):
    d = pd.read_csv(fn, usecols=["Timestamp"]); ts = pd.to_datetime(d["Timestamp"], format="ISO8601")
    n_fl += len(d); secs += (ts.max() - ts.min()).total_seconds()
rate_h = n_fl / secs * 3600
M("flowRate", f"{n_fl / secs:.2f}"); M("flowsPerHour", intc(int(round(rate_h, -2))))
M("alertsOrig", intc(int(round(rate_h * 0.55, -2))))
M("alertsPnnA", intc(int(round(rate_h * S["A"]["p99"]["LSTM-AE"]["fpr"][0], -1))))
M("fprRatioA", f"{S['A']['p99']['LSTM-AE']['fpr_attack_capture_background'][0] / S['A']['p99']['LSTM-AE']['fpr_benign_captures'][0]:.0f}")
M("alertsPnnBgA", intc(int(round(rate_h * S["A"]["p99"]["LSTM-AE"]["fpr_attack_capture_background"][0], -1))))

# ---------------- cost ----------------
L = S["A"]["f1"]["LSTM-AE"]
M("lstmParams", intc(L["n_params"])); M("denseParams", intc(S["A"]["f1"]["Dense-AE"]["n_params"]))
M("lstmTrainMin", f"{L['train_sec'][0] / 60:.1f}"); M("lstmInferMs", f"{L['inference_ms_per_window'][0]:.2f}")
M("lstmEpochs", f"{L['epochs_run'][0]:.0f}")

with open(os.path.join(OUT, "numbers.tex"), "w") as fh:
    for k, v in sorted(macros.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(macros), "macros;", sorted(os.listdir(os.path.join(OUT, "tables"))))
