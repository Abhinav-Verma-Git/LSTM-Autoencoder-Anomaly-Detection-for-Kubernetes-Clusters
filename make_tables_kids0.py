"""LaTeX tables (tables/k_*.tex) and macros (numbers_k.tex) for the Kube-IDS0 direct replication."""
import json
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "tex"
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
S = json.load(open("results_kids0/summary.json"))
MODELS = ["LSTM-AE", "Dense-AE", "IForest", "PCA"]
mac = {}
num = lambda x, p=3: f"{x:.{p}f}"
intc = lambda n: f"{n:,}".replace(",", "{,}")


def pm(v, p=3, bold=False):
    s = num(v[0], p) if v[1] == 0 else f"\\pmv{{{num(v[0], p)}}}{{{num(v[1], p)}}}"
    return f"\\textbf{{{s}}}" if bold else s


def M(k, v):
    mac[k] = v


def best(vals):
    return int(np.argmax([v[0] for v in vals]))


F, A, B = S["F"], S["A"], S["B"]
# ---------- data ----------
d = F["data"]
M("kRows", intc(d["rows_total"])); M("kAttacks", intc(d["attack_rows_total"])); M("kNormal", intc(d["rows_total"] - d["attack_rows_total"]))
M("kFeat", str(d["n_features_used"])); M("kConcNormalF", intc(d["test_concurrent_normal"]))
M("kPctConc", f"{100 * d['test_concurrent_normal'] / (d['rows_total'] - d['attack_rows_total']):.0f}")
M("kPrevF", f"{100 * F['prevalence']:.1f}"); M("kTrivFoneF", num(F["trivial"]["f1"])); M("kTrivPrecF", num(F["trivial"]["precision"]))
M("kTrainF", intc(d["n_train_windows"])); M("kTrainA", intc(A["data"]["n_train_windows"])); M("kTrainB", intc(B["data"]["n_train_windows"]))
for p in "AB":
    M(f"kPrev{p}", f"{100 * S[p]['prevalence']:.1f}"); M(f"kTrivFone{p}", num(S[p]["trivial"]["f1"]))

# ---------- Table: reproduction (Protocol F) ----------
cols = [("precision_attack", "Prec."), ("recall_attack", "Recall"), ("f1_attack", "F1"), ("recall_normal", "Norm.\\ rec."),
        ("accuracy", "Acc."), ("weighted_f1", "W-F1"), ("roc_auc", "ROC-AUC")]
rows = ["LSTM-AE~\\cite{svyrydov2026} (reported) & 0.73 & 0.95 & 0.83 & 0.45 & 0.76 & 0.74 & -- \\\\", "\\midrule"]
bests = {k: best([F["models"][m]["test_f1_threshold"][k] for m in MODELS]) for k, _ in cols if k not in ("recall_attack",)}
for i, m in enumerate(MODELS):
    t = F["models"][m]["test_f1_threshold"]
    cells = " & ".join(pm(t[k], bold=(bests.get(k) == i)) for k, _ in cols)
    rows.append(f"{m} (ours) & {cells} \\\\")
tr = F["trivial"]
rows += ["\\midrule", f"Flag every flow as attack & {num(tr['precision'])} & 1.000 & {num(tr['f1'])} & 0.000 & {num(tr['accuracy'])} & -- & 0.500 \\\\"]
open(os.path.join(OUT, "tables", "k_repro.tex"), "w").write(
    "\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}\n\\toprule\nDetector & " + " & ".join(c for _, c in cols) + " \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
L = F["models"]["LSTM-AE"]
for k, short in [("precision_attack", "Prec"), ("recall_attack", "Rec"), ("f1_attack", "Fone"), ("recall_normal", "RecN"),
                 ("accuracy", "Acc"), ("weighted_f1", "WFone"), ("roc_auc", "Auc"), ("pr_auc", "Pr"), ("fpr", "Fpr")]:
    M(f"kF{short}", num(L["test_f1_threshold"][k][0])); M(f"kF{short}Sd", num(L["test_f1_threshold"][k][1]))
M("kFPnnRec", num(L["test_p99_threshold"]["recall_attack"][0])); M("kFPnnFpr", num(L["test_p99_threshold"]["fpr"][0]))
for m, key in [("Dense-AE", "Dense"), ("IForest", "If"), ("PCA", "Pca")]:
    t = F["models"][m]["test_f1_threshold"]
    M(f"kF{key}Fone", num(t["f1_attack"][0])); M(f"kF{key}Auc", num(t["roc_auc"][0])); M(f"kF{key}RecN", num(t["recall_normal"][0]))
    M(f"kF{key}PnnRec", num(F["models"][m]["test_p99_threshold"]["recall_attack"][0]))
fo = [F["models"][m]["test_f1_threshold"]["f1_attack"][0] for m in MODELS]
M("kFFoneBest", num(max(fo))); M("kFFoneBestModel", MODELS[int(np.argmax(fo))])
M("kFFoneMarg", num(L["test_f1_threshold"]["f1_attack"][0] - F["trivial"]["f1"]))
M("kFFoneSeeds", ", ".join(num(x, 3) for x in [r for r in json.load(open("results_kids0/main_F.json"))["runs"] if r["model"] == "LSTM-AE"] and
                         [r["test_f1_threshold"]["f1_attack"] for r in json.load(open("results_kids0/main_F.json"))["runs"] if r["model"] == "LSTM-AE"]))
pn = [F["models"][m]["test_p99_threshold"]["recall_attack"][0] for m in MODELS]
M("kFPnnRecMax", num(max(pn)))

# ---------- Table: leakage-free protocols A and B ----------
lines = []
for p in "AB":
    P = S[p]["models"]
    keys = [("test_f1_threshold", "roc_auc"), ("test_f1_threshold", "pr_auc"), ("test_f1_threshold", "f1_attack"),
            ("test_f1_threshold", "fpr"), ("test_p99_threshold", "precision_attack"), ("test_p99_threshold", "recall_attack")]
    hi = {k: best([P[m][k[0]][k[1]] for m in MODELS]) for k in keys if k[1] != "fpr"}
    for i, m in enumerate(MODELS):
        cells = " & ".join(pm(P[m][a][b], bold=(hi.get((a, b)) == i)) for a, b in keys)
        pre = f"\\multirow{{4}}{{*}}{{{p}}}" if i == 0 else ""
        lines.append(f"{pre} & {m} & {cells} \\\\")
    if p == "A":
        lines.append("\\midrule")
    L = P["LSTM-AE"]
    M(f"kAuc{p}", num(L["test_f1_threshold"]["roc_auc"][0])); M(f"kPr{p}", num(L["test_f1_threshold"]["pr_auc"][0]))
    M(f"kFone{p}", num(L["test_f1_threshold"]["f1_attack"][0])); M(f"kFpr{p}", num(L["test_f1_threshold"]["fpr"][0]))
    M(f"kRecN{p}", num(L["test_f1_threshold"]["recall_normal"][0]))
    M(f"kPnnPrec{p}", num(L["test_p99_threshold"]["precision_attack"][0])); M(f"kPnnRec{p}", num(L["test_p99_threshold"]["recall_attack"][0]))
    M(f"kPnnFpr{p}", num(L["test_p99_threshold"]["fpr"][0]))
    au = [P[m]["test_f1_threshold"]["roc_auc"][0] for m in MODELS]
    M(f"kAucBest{p}", num(max(au))); M(f"kAucBestModel{p}", MODELS[int(np.argmax(au))])
    M(f"kAucMin{p}", num(min(au))); M(f"kAucMax{p}", num(max(au)))
    rr = [P[m]["test_p99_threshold"]["recall_attack"][0] for m in MODELS]
    M(f"kPnnRecMax{p}", num(max(rr))); M(f"kPnnRecBestModel{p}", MODELS[int(np.argmax(rr))])
open(os.path.join(OUT, "tables", "k_protocols.tex"), "w").write(
    "\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\n & & & & \\multicolumn{2}{c}{$\\theta^{*}$ (F1-optimal)} & \\multicolumn{2}{c}{$\\theta_{99}$ (label-free)} \\\\\n"
    "\\cmidrule(lr){5-6}\\cmidrule(l){7-8}\n & Model & ROC-AUC & PR-AUC & F1 & FPR & Precision & Recall \\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
M("kTestA", intc(A["data"]["test"]["normal"] + A["data"]["test"]["attack"])); M("kTestB", intc(B["data"]["test"]["normal"] + B["data"]["test"]["attack"]))

# ---------- Table: same-moment control ----------
lines = []
for p in "FAB":
    for i, m in enumerate(MODELS):
        x = S[p]["models"][m]["extra"]
        pre = f"\\multirow{{4}}{{*}}{{{p}}}" if i == 0 else ""
        lines.append(f"{pre} & {m} & {pm(x['auc_attack_vs_concurrent_normal'])} & {pm(x['auc_attack_vs_nonconcurrent_normal'])} & "
                     f"{pm(x['auc_concurrent_vs_nonconcurrent_normal'])} \\\\")
    if p != "B":
        lines.append("\\midrule")
    x = S[p]["models"]["LSTM-AE"]["extra"]
    M(f"kConc{p}", num(x["auc_attack_vs_concurrent_normal"][0])); M(f"kNonconc{p}", num(x["auc_attack_vs_nonconcurrent_normal"][0]))
    M(f"kCvN{p}", num(x["auc_concurrent_vs_nonconcurrent_normal"][0]))
    M(f"kFprConcF{p}", num(x["fpr_concurrent_f1"][0])); M(f"kFprNonconcF{p}", num(x["fpr_nonconcurrent_f1"][0]))
    vals = [S[p]["models"][m]["extra"]["auc_attack_vs_nonconcurrent_normal"][0] for m in MODELS]
    M(f"kNonconcMin{p}", num(min(vals))); M(f"kNonconcMax{p}", num(max(vals)))
    vals = [S[p]["models"][m]["extra"]["auc_attack_vs_concurrent_normal"][0] for m in MODELS]
    M(f"kConcMin{p}", num(min(vals))); M(f"kConcMax{p}", num(max(vals)))
    vals = [S[p]["models"][m]["extra"]["auc_concurrent_vs_nonconcurrent_normal"][0] for m in MODELS]
    M(f"kCvNMin{p}", num(min(vals))); M(f"kCvNMax{p}", num(max(vals)))
open(os.path.join(OUT, "tables", "k_control.tex"), "w").write(
    "\\begin{tabular}{@{}llccc@{}}\n\\toprule\n & & (a) Attacks vs & (b) Attacks vs & (c) Benign during vs \\\\\n"
    "Prot. & Model & benign during attack & benign outside & benign outside attacks \\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------- ablation ----------
if "ablation" in S:
    Ab = S["ablation"]
    ref = A["models"]["LSTM-AE"]; refF = F["models"]["LSTM-AE"]
    cfg = [("A", "$T=10$, all features (main)", ref), ("A", "$T=1$, all features", Ab["A|LSTM-AE(T=1)"]),
           ("A", "$T=10$, network only", Ab["A|LSTM-AE(net)"]), ("A", "$T=10$, container only", Ab["A|LSTM-AE(ctr)"]),
           ("F", "$T=10$, all features (main)", refF), ("F", "$T=10$, network only", Ab["F|LSTM-AE(net)"])]
    body = []
    for p, n, dd in cfg:
        t, x = dd["test_f1_threshold"], dd["extra"]
        body.append(f"{p} & {n} & {pm(t['roc_auc'])} & {pm(t['f1_attack'])} & {pm(x['auc_attack_vs_concurrent_normal'])} & {pm(x['auc_attack_vs_nonconcurrent_normal'])} \\\\")
        if p == "A" and n.startswith("$T=10$, container"):
            body.append("\\midrule")
    open(os.path.join(OUT, "tables", "k_ablation.tex"), "w").write(
        "\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n & & & & \\multicolumn{2}{c}{ROC-AUC vs benign} \\\\\n\\cmidrule(l){5-6}\n"
        "Prot. & Variant & ROC-AUC & F1 ($\\theta^{*}$) & during attack & outside \\\\\n\\midrule\n" + "\n".join(body) +
        "\n\\bottomrule\n\\end{tabular}\n")
    for key, tag in [("A|LSTM-AE(T=1)", "AOne"), ("A|LSTM-AE(net)", "ANet"), ("A|LSTM-AE(ctr)", "ACtr"), ("F|LSTM-AE(net)", "FNet")]:
        t, x = Ab[key]["test_f1_threshold"], Ab[key]["extra"]
        M(f"kAbl{tag}Auc", num(t["roc_auc"][0])); M(f"kAbl{tag}Fone", num(t["f1_attack"][0]))
        M(f"kAbl{tag}Conc", num(x["auc_attack_vs_concurrent_normal"][0])); M(f"kAbl{tag}Nonconc", num(x["auc_attack_vs_nonconcurrent_normal"][0]))
        M(f"kAbl{tag}RecN", num(t["recall_normal"][0]))

# ---------- per-scenario (LSTM-AE, Protocol A) ----------
P = A["models"]["LSTM-AE"]
for s, tag in [("1", "DS"), ("2", "DT"), ("3", "BF"), ("4", "SQ"), ("5", "BS"), ("6", "BT")]:
    M(f"kScen{tag}", num(P["per_scenario_auc"][s][0]))
    if s in P["per_scenario_auc_vs_concurrent"] and P["per_scenario_auc_vs_concurrent"][s]:
        M(f"kScenConc{tag}", num(P["per_scenario_auc_vs_concurrent"][s][0]))
L = A["models"]["LSTM-AE"]
M("kParams", intc(L["n_params"]))
pr = L["test_p99_threshold"]["per_scenario_recall"]
for s, tag in [("1", "DS"), ("2", "DT"), ("3", "BF"), ("4", "SQ"), ("5", "BS"), ("6", "BT")]:
    M(f"kPnnRec{tag}", num(pr[s][0]))
nA = A["data"]["test"]["by_scenario"]
det = {s: pr[s][0] * nA[s] for s in pr}
M("kPnnShareSQ", f"{100 * det['4'] / sum(det.values()):.0f}")
dos = [pr[s][0] for s in ["1", "2", "6"]]
M("kPnnRecDoSMin", num(min(dos))); M("kPnnRecDoSMax", num(max(dos)))
prB = B["models"]["LSTM-AE"]["test_p99_threshold"]["per_scenario_recall"]
M("kPnnRecSQB", num(prB["4"][0])); M("kPnnRecOtherMaxB", num(max(v[0] for k, v in prB.items() if k != "4")))
M("kPnnFprA", num(L["test_p99_threshold"]["fpr"][0]))
M("kNonconcBv", num(B["models"]["LSTM-AE"]["extra"]["auc_attack_vs_nonconcurrent_normal"][0]))

with open(os.path.join(OUT, "numbers_k.tex"), "w") as fh:
    for k, v in sorted(mac.items()):
        assert k.isalpha(), k
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(mac), "macros")
