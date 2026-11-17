"""Additional paper figures, drawn ONLY from saved experiment outputs (no retraining, no synthetic data).

Inputs : results_kids0/{summary,main_F,main_A}.json, results_kids0/scores/*.npz, results_kids0/run.log,
         results_v2/summary.json, the Kube-IDS0 cache (kids0.load(), for timestamps).
Checks : before plotting, every ROC-AUC, threshold and confusion-matrix count recomputed from the saved scores is
         asserted to equal the value stored in the result JSON files that the paper's tables are built from.
Usage  : KIDS0_DIR=/path/to/kube-ids0 python make_paper_figs.py figs
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

import kids0

OUT = sys.argv[1] if len(sys.argv) > 1 else "figs"
os.makedirs(OUT, exist_ok=True)
R = "results_kids0"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 8,
    "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0, "savefig.dpi": 300,
})
MC = {"LSTM-AE": "#1f5fbf", "Dense-AE": "#7a7a7a", "PCA": "#d9822b", "IForest": "#2e8b57", "LSTM-AE (T=1)": "#8e44ad"}
MLS = {"LSTM-AE": "-", "Dense-AE": "--", "PCA": "-.", "IForest": ":", "LSTM-AE (T=1)": (0, (5, 1.5))}
SC = {0: "#b5b5b5", 1: "#1f77b4", 2: "#17becf", 3: "#d62728", 4: "#9467bd", 5: "#ff7f0e", 6: "#8c564b"}
SN = {0: "Benign", 1: "DVWA Slowloris", 2: "DVWA Tor's Hammer", 3: "DVWA brute force", 4: "DVWA SQL injection",
      5: "BoA Slowloris", 6: "BoA Tor's Hammer"}
S = json.load(open(f"{R}/summary.json"))


def run(p, model, seed=0):
    return [r for r in json.load(open(f"{R}/main_{p}.json"))["runs"] if r["model"] == model and r["seed"] == seed][0]


def scores(p, model, seed=0):
    return np.load(f"{R}/scores/{p}_{model}_s{seed}.npz")["st"]


def close(a, b, tol=1e-4, what=""):
    assert abs(a - b) <= tol, f"MISMATCH {what}: recomputed {a} vs stored {b}"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- consistency checks against stored results
LF = np.load(f"{R}/scores/F_labels.npz"); LA = np.load(f"{R}/scores/A_labels.npz")
for p, L in [("F", LF), ("A", LA)]:
    for m in ["LSTM-AE", "Dense-AE", "PCA", "IForest"]:
        close(roc_auc_score(L["y"], scores(p, m)), run(p, m)["test_f1_threshold"]["roc_auc"], what=f"AUC {p} {m}")
close(roc_auc_score(LA["y"], scores("A", "LSTM-AE(T=1)")),
      [r for r in json.load(open(f"{R}/ablation.json"))["runs"] if r["model"] == "LSTM-AE(T=1)" and r["seed"] == 0][0]
      ["test_f1_threshold"]["roc_auc"], what="AUC T=1")
for p, L in [("F", LF), ("A", LA)]:
    st = scores(p, "LSTM-AE")
    for key in ["test_f1_threshold", "test_p99_threshold"]:
        r = run(p, "LSTM-AE")[key]; pred = st >= r["threshold"]; y = L["y"] == 1
        for k, v in [("tp", pred & y), ("fp", pred & ~y), ("tn", ~pred & ~y), ("fn", ~pred & y)]:
            assert int(v.sum()) == r[k], f"MISMATCH confusion {p} {key} {k}: {int(v.sum())} vs {r[k]}"
print("all recomputed metrics match the stored results")

# ---------------------------------------------------------------- Fig: session timelines (traffic + LSTM-AE score)
df = kids0.load()
assert (LF["s"] == df.scen.to_numpy()).all(), "Protocol F scores are not aligned with the dataset rows"
stF = scores("F", "LSTM-AE"); thF = run("F", "LSTM-AE")["test_f1_threshold"]["threshold"]
th99F = run("F", "LSTM-AE")["test_p99_threshold"]["threshold"]
sess = [("boa-10-25", "BoA, 25 Oct 2024"), ("boa-10-31", "BoA, 31 Oct 2024"),
        ("dvwa-11-18", "DVWA, 18 Nov 2024"), ("dvwa-11-19", "DVWA, 19 Nov 2024")]
spans = [(df.ts[df.session == s].max() - df.ts[df.session == s].min()).total_seconds() / 60 for s, _ in sess]
fig, axs = plt.subplots(2, 4, figsize=(7.16, 3.0), gridspec_kw={"width_ratios": [max(w, 6) for w in spans],
                                                              "height_ratios": [1, 1.25]}, sharex="col")
for j, (s, title) in enumerate(sess):
    m = (df.session == s).to_numpy()
    t = (df.ts[m] - df.ts[m].min()).dt.total_seconds().to_numpy() / 60
    sc = df.scen.to_numpy()[m]
    bins = np.arange(0, t.max() + 1 / 6, 1 / 6)                      # 10-second bins
    present = [k for k in SC if (sc == k).any()]
    axs[0, j].hist([t[sc == k] for k in present], bins=bins, stacked=True, color=[SC[k] for k in present], lw=0)
    axs[0, j].set_title(f"{title} ({m.sum():,} flows)", fontsize=7, pad=2)
    order = np.argsort(sc != 0, kind="stable")                         # benign first, attacks on top
    axs[1, j].scatter(t[order], np.clip(stF[m][order], 1e-3, None), s=0.6, c=[SC[k] for k in sc[order]], lw=0,
                      rasterized=True)
    axs[1, j].axhline(thF, c="k", ls="--", lw=0.7); axs[1, j].axhline(th99F, c="k", ls=":", lw=0.8)
    axs[1, j].set_yscale("log"); axs[1, j].set_ylim(1e-3, 3e4); axs[1, j].set_xlabel("Minutes from session start")
    if j:
        axs[1, j].set_yticklabels([])
axs[0, 0].set_ylabel("Flows / 10 s"); axs[1, 0].set_ylabel(r"LSTM-AE error $\varepsilon$")
h = [plt.Rectangle((0, 0), 1, 1, color=SC[k]) for k in SC] + \
    [plt.Line2D([], [], c="k", ls="--", lw=0.7), plt.Line2D([], [], c="k", ls=":", lw=0.8)]
fig.legend(h, [SN[k] for k in SC] + [r"$\theta^{*}$ (F1-optimal)", r"$\theta_{99}$"], ncol=9, loc="lower center",
           bbox_to_anchor=(0.5, 0.99), frameon=False, fontsize=6.2, handlelength=1.2, columnspacing=0.8)
fig.tight_layout(pad=0.2, w_pad=0.6, h_pad=0.4)
save(fig, "fig_k_timeline.pdf")

# ---------------------------------------------------------------- Fig: Protocol F diagnostics
c, y = LF["c"].astype(bool), LF["y"] == 1
fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.05), gridspec_kw={"width_ratios": [1.35, 1.35, 0.8, 0.8]})
ax = axs[0]
bins = np.logspace(-3, 4.5, 46)
for msk, col, lab, ht in [(y, MC["LSTM-AE"], "Attack flows", "stepfilled"),
                          (~y & c, "#404040", "Benign, during an attack", "step"),
                          (~y & ~c, "#d9822b", "Benign, outside attacks", "step")]:
    ax.hist(np.clip(stF[msk], 1e-3, 3e4), bins=bins, weights=np.full(msk.sum(), 100 / msk.sum()), histtype=ht,
            color=col, alpha=0.35 if ht == "stepfilled" else 1, lw=1.0, label=f"{lab} ({msk.sum():,})")
top = ax.get_ylim()[1]; ax.set_ylim(0, top * 1.55)
ax.vlines(thF, 0, top * 1.1, colors="k", linestyles="--", lw=0.7); ax.vlines(th99F, 0, top * 1.1, colors="k", linestyles=":", lw=0.8)
ax.text(thF * 1.3, top * 1.02, r"$\theta^{*}$", fontsize=7); ax.text(th99F * 1.3, top * 1.02, r"$\theta_{99}$", fontsize=7)
ax.set_xscale("log"); ax.set_xlabel(r"Reconstruction error $\varepsilon$"); ax.set_ylabel("Flows per bin (%)")
ax.legend(frameon=False, loc="upper left", fontsize=5.6, handlelength=1.2, borderaxespad=0.1)
ax.set_title("(a) Error distribution", fontsize=7.5, pad=2)
ax = axs[1]
th = np.unique(np.quantile(stF, np.linspace(0, 1, 2000)))      # same candidate grid as metrics.f1_optimal_threshold
pos = y.sum(); tp = np.array([(stF[y] >= t).sum() for t in th]); fp = np.array([(stF[~y] >= t).sum() for t in th])
prec = tp / np.maximum(tp + fp, 1); rec = tp / pos; f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12); fpr = fp / (~y).sum()
for v, col, ls, lab in [(prec, "#2e8b57", "-", "Precision"), (rec, MC["LSTM-AE"], "-", "Recall"),
                        (f1, "k", "-", "F1"), (fpr, "#d62728", "--", "FPR")]:
    ax.plot(th, v, c=col, ls=ls, lw=1.0, label=lab)
prev = y.mean(); ax.axhline(2 * prev / (1 + prev), c=MC["Dense-AE"], ls=":", lw=0.8, label="Flag-all F1")
ax.vlines(thF, 0, 1.03, colors="k", linestyles="--", lw=0.7)
ax.set_xscale("log"); ax.set_xlim(th[1], th[-1]); ax.set_ylim(0, 1.42); ax.set_yticks([0, .25, .5, .75, 1])
ax.set_xlabel(r"Threshold $\theta$")
ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=5.6, handlelength=1.4, borderaxespad=0.1, columnspacing=0.7)
ax.set_title("(b) Metrics vs threshold", fontsize=7.5, pad=2)
close(f1.max(), run("F", "LSTM-AE")["test_f1_threshold"]["f1_attack"], 1e-6, "F1 max sweep")
close(th[np.argmax(f1)], thF, 1e-9, "theta* from sweep")
for ax, key, title in [(axs[2], "test_f1_threshold", r"(c) At $\theta^{*}$"), (axs[3], "test_p99_threshold", r"(d) At $\theta_{99}$")]:
    r = run("F", "LSTM-AE")[key]
    M = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
    ax.imshow(M / M.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for k in range(2):
            frac = M[i, k] / M[i].sum()
            ax.text(k, i, f"{M[i, k]:,}\n({100 * frac:.1f}%)", ha="center", va="center", fontsize=6.2,
                    color="white" if frac > 0.6 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Attack"]); ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Attack"] if key == "test_f1_threshold" else ["", ""])
    ax.set_xlabel("Predicted"); ax.set_title(title, fontsize=7.5, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
axs[2].set_ylabel("True")
fig.tight_layout(pad=0.2, w_pad=0.8)
save(fig, "fig_k_protF.pdf")

# ---------------------------------------------------------------- Fig: ROC and PR curves, Protocol A
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.95))
ya = LA["y"]
for m, key in [("LSTM-AE", "LSTM-AE"), ("Dense-AE", "Dense-AE"), ("PCA", "PCA"), ("IForest", "IForest"),
               ("LSTM-AE (T=1)", "LSTM-AE(T=1)")]:
    st = scores("A", key)
    f, t, _ = roc_curve(ya, st); pr, rc, _ = precision_recall_curve(ya, st)
    axs[0].plot(f, t, c=MC[m], ls=MLS[m], lw=1.0, label=f"{m} ({roc_auc_score(ya, st):.3f})")
    axs[1].plot(rc, pr, c=MC[m], ls=MLS[m], lw=1.0, label=f"{average_precision_score(ya, st):.3f}")
axs[0].plot([0, 1], [0, 1], c="#c0c0c0", lw=0.6)
axs[1].axhline(ya.mean(), c="#c0c0c0", lw=0.6)
axs[0].set_xlabel("False-positive rate"); axs[0].set_ylabel("True-positive rate")
axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[1].set_ylim(0, 1.02)
axs[0].legend(frameon=False, loc="lower right", fontsize=5.4, handlelength=1.6, borderaxespad=0)
axs[1].legend(frameon=False, loc="lower left", fontsize=5.4, handlelength=1.6, borderaxespad=0, title="PR-AUC",
              title_fontsize=5.6)
axs[0].set_title("(a) ROC", fontsize=7.5, pad=2); axs[1].set_title("(b) Precision-recall", fontsize=7.5, pad=2)
fig.tight_layout(pad=0.2, w_pad=0.8)
save(fig, "fig_k_rocpr.pdf")

# ---------------------------------------------------------------- Fig: training curves
log = open(f"{R}/run.log").read().split("===== ABLATION")[0]
curves = {}
for blk in re.split(r"(?=^Training )", log, flags=re.M):
    mh = re.match(r"Training (\w) (\S+) seed (\d)", blk)
    if mh:
        v = [(float(a), float(b)) for a, b in re.findall(r"ep \d+ train ([\d.]+) val ([\d.]+)", blk)]
        curves[(mh[1], mh[2], int(mh[3]))] = np.array(v)
for p in "FA":
    for m in ["LSTM-AE", "Dense-AE"]:
        for sd in range(5):
            n_ep = len(curves[(p, m, sd)])
            assert n_ep == run(p, m, sd)["epochs_run"], f"log/epochs mismatch {p} {m} {sd}"
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.8), sharey=True)
for ax, p, title in [(axs[0], "F", "(a) Protocol F"), (axs[1], "A", "(b) Protocol A")]:
    for m in ["LSTM-AE", "Dense-AE"]:
        for sd in range(5):
            v = curves[(p, m, sd)]; ep = np.arange(1, len(v) + 1)
            ax.plot(ep, v[:, 1], c=MC[m], lw=0.8, alpha=0.9, label=f"{m}, validation" if sd == 0 else None)
            if sd == 0:
                ax.plot(ep, v[:, 0], c=MC[m], lw=0.8, ls=":", label=f"{m}, training (seed 0)")
    ax.set_xlabel("Epoch"); ax.set_title(title, fontsize=7.5, pad=2); ax.set_xlim(0, 61)
axs[0].set_ylabel("MSE loss (normal windows)")
axs[1].legend(frameon=False, fontsize=5.4, loc="upper right", handlelength=1.6, borderaxespad=0)
fig.tight_layout(pad=0.2, w_pad=0.6)
save(fig, "fig_k_training.pdf")

# ---------------------------------------------------------------- Fig: ROC-AUC across protocols and datasets
V = json.load(open("results_v2/summary.json"))
groups = [("Kube-IDS0\nF", lambda m: S["F"]["models"][m]["test_f1_threshold"]["roc_auc"]),
          ("Kube-IDS0\nA", lambda m: S["A"]["models"][m]["test_f1_threshold"]["roc_auc"]),
          ("Kube-IDS0\nB", lambda m: S["B"]["models"][m]["test_f1_threshold"]["roc_auc"]),
          ("Sever-Dogan\nA", lambda m: V["A"]["f1"][m]["roc_auc"]),
          ("Sever-Dogan\nB", lambda m: V["B"]["f1"][m]["roc_auc"])]
models = ["LSTM-AE", "Dense-AE", "PCA", "IForest"]
hatch = {"LSTM-AE": "", "Dense-AE": "////", "PCA": "....", "IForest": "xxxx"}
fig, ax = plt.subplots(figsize=(3.5, 1.95))
w = 0.2
for i, m in enumerate(models):
    vals = [g(m) for _, g in groups]
    ax.bar(np.arange(len(groups)) + (i - 1.5) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals],
           color=MC[m], alpha=0.85, hatch=hatch[m], edgecolor="white", lw=0, error_kw={"lw": 0.6, "capsize": 1.2},
           label=m)
ax.axhline(0.5, c="k", ls="--", lw=0.6); ax.text(4.62, 0.51, "chance", fontsize=6, ha="right")
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g for g, _ in groups], fontsize=6.6)
ax.set_ylim(0, 1); ax.set_ylabel("Test ROC-AUC")
ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.98), fontsize=6.2, handlelength=1.4)
fig.tight_layout(pad=0.2)
save(fig, "fig_auc_compare.pdf")

# ---------------------------------------------------------------- Fig: window length on both datasets
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.85), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axs[0]
ref = S["A"]["models"]["LSTM-AE"]; one = S["ablation"]["A|LSTM-AE(T=1)"]
mets = [("ROC\nAUC", lambda d: d["test_f1_threshold"]["roc_auc"]), ("PR\nAUC", lambda d: d["test_f1_threshold"]["pr_auc"]),
        ("F1", lambda d: d["test_f1_threshold"]["f1_attack"]),
        ("AUC vs\nconcur.", lambda d: d["extra"]["auc_attack_vs_concurrent_normal"])]
x = np.arange(len(mets))
for dx, d, lab, col in [(-0.18, ref, "$T=10$ (article)", MC["LSTM-AE"]), (0.18, one, "$T=1$", MC["LSTM-AE (T=1)"])]:
    ax.bar(x + dx, [f(d)[0] for _, f in mets], 0.36, yerr=[f(d)[1] for _, f in mets], color=col, alpha=0.85,
           error_kw={"lw": 0.6, "capsize": 1.2}, label=lab)
ax.set_xticks(x); ax.set_xticklabels([n for n, _ in mets], fontsize=6.2); ax.set_ylim(0.5, 1.13)
ax.set_yticks([.5, .6, .7, .8, .9, 1]); ax.set_ylabel("Score")
ax.legend(frameon=False, fontsize=6, loc="upper center", ncol=2, borderaxespad=0, columnspacing=0.8)
ax.set_title("(a) Kube-IDS0, Protocol A", fontsize=7.5, pad=2)
ax = axs[1]
A = V["ablation"]["f1"]; refv = V["ablation_T10_ref"]
get = lambda T: refv if T == 10 else A[f"LSTM-AE(T={T})"]
Ts = [1, 5, 10, 20]
for key, col, lab, mk in [("roc_auc", "k", "All attacks", "s"), ("roc_auc_excl_recon", MC["LSTM-AE"], "Exploits", "o"),
                          ("roc_auc_recon", MC["PCA"], "Recon.", "^")]:
    ax.errorbar(Ts, [get(T)[key][0] for T in Ts], yerr=[get(T)[key][1] for T in Ts], marker=mk, ms=3, c=col, lw=0.9,
                capsize=1.2, label=lab)
ax.axhline(0.5, c="#c0c0c0", ls="--", lw=0.6)
ax.set_xscale("log"); ax.set_xticks(Ts); ax.set_xticklabels([str(t) for t in Ts]); ax.minorticks_off()
ax.set_xlabel("Window length $T$"); ax.set_ylabel("ROC-AUC"); ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=5.8, loc="lower left", borderaxespad=0, handlelength=1.4)
ax.set_title("(b) Sever-Dogan, Protocol A", fontsize=7.5, pad=2)
fig.tight_layout(pad=0.2, w_pad=0.8)
save(fig, "fig_window_both.pdf")
print("done")
