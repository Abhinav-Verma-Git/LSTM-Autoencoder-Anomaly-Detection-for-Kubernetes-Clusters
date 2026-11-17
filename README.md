# LSTM Autoencoder Anomaly Detection for Kubernetes — reimplementation and cross-dataset evaluation

Cloud SRE and Security course project, Department of Computer Science and Engineering, IIIT Dharwad.
Authors: Yogesh Gupta (23BCS137), Abhinav Verma (23BCS004), Parmitha Vangapandu (23BCS092).
Supervisor: Prof. Malay Kumar.

Reimplements and evaluates:
A. Svyrydov, O. Sievierinov, "A Method for Anomaly Detection in Kubernetes Clusters Using an LSTM Autoencoder",
Herald of Khmelnytskyi National University, Technical Sciences, 361(1), 501–509, 2026. doi:10.31891/2307-5732-2026-361-70

## What is not in this repository

The datasets are not included (they are large and have their own licences). Download them as described below:
Kube-IDS0 CSVs go in `data/kube-ids0/`, the Sever–Dogan dataset in `data_k8s/`. `.gitignore` keeps both folders,
the cached `kids0_cache.pkl` and your virtual environment out of Git. Everything needed to run the code, plus the
saved results behind every table and figure of the paper (`results_kids0/`, `results_v2/`, `results_relabel/`), is
included.

## Live demo (presentation)

See **RUN_IN_VSCODE.md**. In short: `pip install -r requirements.txt`, put the Kube-IDS0 CSVs in `data/kube-ids0/`,
then run `python demo.py` (about 5 minutes), `python demo.py --fast`, `python demo.py --relabel` or `python demo.py --saved`.

## Code provenance

The original article releases no source code and has no code/data availability statement, and no public repository
by its authors was found. **All code here was written by us from the article's equations and text.** Details the
article does not specify (layer sizes, optimiser, epochs, batch size, split procedure, stride, window labelling) are
our choices and are listed in Table I of the paper.

## Direct replication on Kube-IDS0 (the original article's dataset)

Download Kube-IDS0 from Kaggle (https://www.kaggle.com/datasets/redamorsli/kube-ids0) and place, under one folder
(default `data/kube-ids0/`, any sub-folders; or set the `KIDS0_DIR` environment variable):
`dvwa_dataset_ml_ready.csv`, `boa_dataset_ml_ready_frontend_microservice.csv` (features + labels; together exactly the
article's 42,705 attack samples) and the two `*_merged_not_cleaned*.csv` files (used only to recover timestamps).

```
python run_kids0.py --part main --seeds 5      # Protocols F (paper-faithful), A (blocked), B (chronological)
python run_kids0.py --part ablation --seeds 3  # T=1, network-only, container-only
python analyze_kids0.py && python make_tables_kids0.py tex && python make_figs_kids0.py figs
```

| File | Purpose |
|---|---|
| `kids0.py` | loads the ML-ready files, recovers timestamps by order-preserving alignment, builds Protocols F/A/B |
| `run_kids0.py` | experiments incl. the concurrent-benign control (attacks vs benign flows recorded during the attack) |
| `analyze_kids0.py`, `make_tables_kids0.py`, `make_figs_kids0.py` | aggregation, LaTeX tables/macros, figure |

Headline (LSTM-AE, mean of 5 seeds): Protocol F precision 0.755, recall 0.975, F1 0.851 (article: 0.73/0.95/0.83);
flag-everything F1 0.762; PCA and Dense-AE F1 0.890; Protocol A ROC-AUC 0.845 (Dense-AE 0.871); T=1 ROC-AUC 0.980.

## Cross-dataset data

For the cross-dataset evaluation we use the public Kubernetes misuse dataset of Sever & Dogan (2023):

```
git clone https://github.com/yigitsever/kubernetes-dataset data_k8s
```

### Flow-level relabelling (`relabel.py`)

The dataset labels every flow of an attack capture as malicious, including the cluster's control-plane traffic
recorded during the attack. Attacker IPs are not visible (Kube-OVN source-NATs external traffic to 100.64.0.2), so a
flow in an attack capture is labelled malicious only if:

* **R1** it is an inbound request from outside the cluster to the scenario's target service
  (Grafana :3000 for scenarios 1, 5; Node-RED :1880 for 2, 3, 4; InfluxDB :8086 for 6), or
* **R2** (scenarios 3, 4) it is to/from the compromised Node-RED pod (reverse shell, tool downloads, API calls).

Everything else becomes benign. Result: 1,881 of 4,128 attack-labelled flows (45.6%) are control-plane traffic;
scenarios 7–10 (runc, containerd, kubelet CVEs) have no network-visible attack flow. `python relabel.py data_k8s`
prints the per-scenario counts (`results_relabel/relabel_counts.json`).

## Files

| File | Purpose |
|---|---|
| `relabel.py` | flow-level relabelling rules R1/R2 |
| `data.py` | loading, imputation + standardisation fitted on normal training rows, causal windows (one per flow), Protocol A (20-flow blocks alternate val/test) and B (chronological halves) splits (`build_flow_splits`) |
| `models.py` | LSTM autoencoder, dense autoencoder, PCA and Isolation Forest baselines, training with early stopping, reconstruction error |
| `metrics.py` | F1-optimal threshold, 99th-percentile threshold, metrics incl. exploit/recon AUC and the same-capture session-shift control |
| `run_v2.py` | `--part main` (both protocols, 5 seeds), `capture` (original labels), `ablation` (T=1/5/20, log scaling), `extra` (200 epochs; mean/max/last-step scores) |
| `analyze_v2.py` | aggregates seeds into `results_v2/summary.json` and `summary.md` |
| `make_tables.py` | writes every LaTeX table and in-text number of the paper from `summary.json` |
| `make_ieee_figs_v2.py` | paper figures |
| `make_paper_figs.py` | Kube-IDS0 timelines, Protocol F diagnostics, ROC/PR, training curves, cross-dataset and window-length figures; recomputes metrics from saved scores and asserts they match the stored results (`KIDS0_DIR=... python make_paper_figs.py figs`) |

## Reproduce

```
pip install -r requirements.txt
python relabel.py data_k8s
python run_v2.py --part main --seeds 5
python run_v2.py --part capture --seeds 3
python run_v2.py --part ablation --seeds 3
python run_v2.py --part extra --seeds 3
python analyze_v2.py && python make_tables.py tex && python make_ieee_figs_v2.py figs
```

CPU only; the full run takes about 2.5 hours on 2 cores. Per-epoch logs: `results_v2/run.log`.
