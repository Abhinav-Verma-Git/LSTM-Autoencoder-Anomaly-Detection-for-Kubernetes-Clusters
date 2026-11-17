"""Kube-IDS0 figures: per-attack ROC-AUC (vs all benign and vs benign recorded during the attack)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "figs"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 8,
                     "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
BLUE, DARK, GREY, ORANGE = "#1f5fbf", "#404040", "#9a9a9a", "#d9822b"
S = json.load(open("results_kids0/summary.json"))
names = {"1": "DVWA DoS (Slowloris)", "2": "DVWA DoS (Tor's Hammer)", "3": "DVWA brute force", "4": "DVWA SQL injection",
         "5": "BoA DoS (Slowloris)", "6": "BoA DoS (Tor's Hammer)"}
P = S["A"]["models"]
fig, ax = plt.subplots(figsize=(3.5, 2.35))
order = sorted(names, key=lambda k: P["LSTM-AE"]["per_scenario_auc"][k][0])
for i, k in enumerate(order):
    for m, col, dy in [("LSTM-AE", BLUE, 0.13), ("Dense-AE", GREY, -0.13)]:
        a = P[m]["per_scenario_auc"][k]
        ax.errorbar(a[0], i + dy, xerr=a[1], fmt="o", ms=4, color=col, lw=0.6)
        c = P[m]["per_scenario_auc_vs_concurrent"].get(k)
        if c:
            ax.plot([c[0], a[0]], [i + dy, i + dy], color=col, lw=0.6, alpha=0.6)
            ax.plot(c[0], i + dy, "o", ms=4, mfc="white", mec=col, mew=1)
ax.axvline(0.5, ls="--", c=GREY, lw=0.6)
ax.set_yticks(range(len(order))); ax.set_yticklabels([names[k] for k in order], fontsize=6.8)
ax.set_xlim(0.3, 1.0); ax.set_xlabel("ROC-AUC (Kube-IDS0, Protocol A, 5 seeds)")
ax.plot([], [], "o", c=BLUE, ms=4, label="LSTM-AE vs all benign"); ax.plot([], [], "o", mfc="white", mec=BLUE, ms=4, label="LSTM-AE vs benign during attack")
ax.plot([], [], "o", c=GREY, ms=4, label="Dense-AE vs all benign"); ax.plot([], [], "o", mfc="white", mec=GREY, ms=4, label="Dense-AE vs benign during attack")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2, fontsize=6.3, handletextpad=0.2, columnspacing=0.8)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(OUT, "fig_k_scenario.pdf")); plt.close(fig)
print("ok")
