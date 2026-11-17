"""Vector figures (IEEE column width 3.5 in) for the relabelled (v2) results."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

R = "results_v2"
OUT = sys.argv[1] if len(sys.argv) > 1 else "figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
})
BLUE, GREY, ORANGE, DARK, LIGHT = "#1f5fbf", "#9a9a9a", "#d9822b", "#404040", "#c8c8c8"
S = json.load(open(os.path.join(R, "summary.json")))

# Fig: composition of attack captures after relabelling
rc = sorted(json.load(open("results_relabel/relabel_counts.json")), key=lambda r: r["flows"])
short = {1: "Grafana SSRF (1)", 2: "Node-RED recon (2)", 3: "Node-RED RCE (3)", 4: "Container escape (4)",
         5: "Grafana path trav. (5)", 6: "InfluxDB auth bypass (6)", 7: "runc symlink race (7)",
         8: "kubelet subPath (8)", 9: "containerd (9)", 10: "runc overwrite (10)"}
fig, ax = plt.subplots(figsize=(3.5, 2.3))
y = np.arange(len(rc))
r1 = np.array([r["attack_R1"] for r in rc]); r2 = np.array([r["attack_R2"] for r in rc]); bg = np.array([r["relabelled_benign"] for r in rc])
ax.barh(y, r1, color=BLUE, label="Attack: request to target service (R1)")
ax.barh(y, r2, left=r1, color=ORANGE, label="Attack: compromised-pod traffic (R2)")
ax.barh(y, bg, left=r1 + r2, color=LIGHT, label="Relabelled benign (control plane)")
for i, r in enumerate(rc):
    ax.text(r["flows"] * 1.08, i, f"{r['attack']}/{r['flows']}", va="center", fontsize=6.5, color=DARK)
ax.set_xscale("log"); ax.set_xlim(1, 9000)
ax.set_yticks(y); ax.set_yticklabels([short[r["scenario"]] for r in rc], fontsize=6.5)
ax.set_xlabel("Flows in attack capture (log scale)")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.42, 1.0), fontsize=6.5, ncol=1, handlelength=1.2)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig_relabel.pdf")); plt.close(fig)

# Fig: ROC curves, exploits-only vs reconnaissance-only (Protocol A, seed 0)
z = np.load(os.path.join(R, "scores_seed0_A.npz"))
yt, st = z["y_test"], z["s_test"]
fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.0), sharey=True)
for ax, (title, mask) in zip(axs, [("Exploits (scen. 3-6)", st != 2), ("Reconnaissance (scen. 2)", (st == 0) | (st == 2))]):
    for m, c, lw in [("LSTM-AE", BLUE, 1.3), ("Dense-AE", GREY, 0.9), ("IForest", DARK, 0.9), ("PCA", LIGHT, 0.9)]:
        f, t, _ = roc_curve(yt[mask], z[f"{m}__test"][mask]); ax.plot(f, t, c=c, lw=lw, label=m, drawstyle="steps-post")
    ax.plot([0, 1], [0, 1], ":", c=GREY, lw=0.6); ax.set_title(title, fontsize=8, pad=2)
    ax.set_xlabel("FPR"); ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
axs[0].set_ylabel("TPR"); axs[1].legend(frameon=False, loc="lower right", fontsize=6, handlelength=1.2)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig_roc.pdf")); plt.close(fig)

# Fig: reconstruction-error distribution (Protocol A, seed 0)
e, c = z["LSTM-AE__test"], z["c_test"]
r0 = [r for r in json.load(open(os.path.join(R, "main_A.json")))["runs"] if r["model"] == "LSTM-AE"][0]
th1, th2 = r0["test_f1_threshold"]["threshold"], r0["test_p99_threshold"]["threshold"]
bins = np.logspace(-3, 4, 29)
groups = [((yt == 0) & (c == 0), GREY, "Benign captures", "stepfilled"),
          ((yt == 0) & (c > 0), DARK, "Benign flows in attack captures", "step"),
          ((yt == 1) & (st != 2), BLUE, "Exploit flows", "step"),
          (st == 2, ORANGE, "Reconnaissance flows", "step")]
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for msk, col, lab, ht in groups:
    ax.hist(np.clip(e[msk], 1e-3, 9.9e3), bins=bins, weights=np.full(msk.sum(), 100 / msk.sum()), histtype=ht,
            color=col, alpha=0.35 if ht == "stepfilled" else 1, lw=1.1, label=f"{lab} (n = {msk.sum():,})",
            ls="--" if col == DARK else "-")
ax.axvline(th1, c=DARK, ls="--", lw=0.8, label=r"$\theta_{F1}=%.2f$" % th1)
ax.axvline(th2, c=DARK, ls=":", lw=0.9, label=r"$\theta_{99}=%.2f$" % th2)
ax.set_xscale("log"); ax.set_xlim(1e-3, 1e4)
ax.set_xlabel(r"Reconstruction error $\varepsilon$ (log scale)"); ax.set_ylabel("Flows per bin (%)")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.3, handlelength=1.5, columnspacing=0.8)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig_error_dist.pdf")); plt.close(fig)

# Fig: window length (ablation)
if "ablation" in S:
    A = S["ablation"]["f1"]; ref = S["ablation_T10_ref"]
    get = lambda T: ref if T == 10 else A[f"LSTM-AE(T={T})"]
    Ts = [1, 5, 10, 20]
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    for key, col, lab, mk in [("roc_auc_excl_recon", BLUE, "Exploits", "o"), ("roc_auc_recon", ORANGE, "Reconnaissance", "^"),
                              ("roc_auc", DARK, "All attacks", "s")]:
        ax.errorbar(Ts, [get(T)[key][0] for T in Ts], yerr=[get(T)[key][1] for T in Ts], marker=mk, ms=3.5, c=col,
                    lw=1, capsize=1.5, label=lab)
    ax.axhline(0.5, c=GREY, ls="--", lw=0.6)
    ax.set_xscale("log"); ax.set_xticks(Ts); ax.set_xticklabels([str(t) for t in Ts]); ax.minorticks_off()
    ax.set_xlabel("Window length $T$ (flows)"); ax.set_ylabel("ROC-AUC")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=1.4)
    fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig_window.pdf")); plt.close(fig)
print("ok")
