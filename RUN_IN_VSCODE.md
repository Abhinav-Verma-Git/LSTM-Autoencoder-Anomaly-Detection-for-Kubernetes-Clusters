# Running the live demo in VS Code

Do steps 1–6 **before** the presentation (installing PyTorch takes a few minutes). During the presentation you only
run step 7.

## 1. Install Python

Install **Python 3.12** (or 3.11) from https://www.python.org/downloads/.
On Windows, tick **"Add python.exe to PATH"** on the first installer screen.

## 2. Install the VS Code Python extension

In VS Code: Extensions (Ctrl+Shift+X) → search **Python** → install the one by Microsoft.

## 3. Open the right folder

File → Open Folder → choose the **`k8s-lstm-ae`** folder (the one that contains `demo.py`, not its parent).
Then open a terminal: Terminal → New Terminal.

## 4. Create a virtual environment and install the packages

Windows (PowerShell, the default VS Code terminal):

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You should now see `(.venv)` at the start of the terminal line. If VS Code asks "select interpreter", pick the one in
`.venv`.

* If PowerShell says *running scripts is disabled*, run
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, then activate again.
* If `python` is not found on Windows, use `py` instead (`py -m venv .venv`).

## 5. Put the Kube-IDS0 data in place

Copy your unzipped Kube-IDS0 files into **`data/kube-ids0/`** (sub-folders are fine; unzip any inner zip files).
These four CSV files must be somewhere inside it:

* `dvwa_dataset_ml_ready.csv`
* `boa_dataset_ml_ready_frontend_microservice.csv`
* `dvwa_merged_not_cleaned.csv`
* `boa_merged_not_cleaned_frontend_microservice.csv`

## 6. (Optional) Sever–Dogan data, for the label-audit demo

On https://github.com/yigitsever/kubernetes-dataset click **Code → Download ZIP**, unzip it, and rename the folder
to **`data_k8s`** inside `k8s-lstm-ae`, so that `data_k8s/benign` and `data_k8s/malicious` exist.
(Or, with git installed: `git clone https://github.com/yigitsever/kubernetes-dataset data_k8s`.)

## 7. Run the demo

| Command | What it shows | Time |
|---|---|---|
| `python demo.py` | Full live demo: loads Kube-IDS0, trains the LSTM autoencoder, reproduces the paper's numbers, compares with "flag everything" and a dense autoencoder, then windows (T=10) vs single flows (T=1) | about 4–6 min |
| `python demo.py --fast` | Same steps with shorter training | about 1.5–2 min |
| `python demo.py --relabel` | Sever–Dogan label audit: 1,881 of 4,128 "attack" flows are normal traffic | seconds |
| `python demo.py --saved` | Prints the saved results of the full experiments (5 seeds) without training | instant |

Rehearse once before the presentation: the first run also builds a cache (`data/kube-ids0/kids0_cache.pkl`), so later
runs start faster. Keep `python demo.py --saved` as a backup if anything goes wrong live.

Live numbers differ slightly from the paper's tables, because the paper reports the mean of 5 seeds and training
varies a little between computers. The conclusions stay the same.

## Troubleshooting

* **`FileNotFoundError: Expected exactly one CSV containing ...`**: the Kube-IDS0 files are not inside
  `data/kube-ids0`, or a file is still zipped, or there are two copies of the same file.
* **`ModuleNotFoundError: No module named 'torch'`**: the virtual environment is not active. Run the activate command
  from step 4 again (you should see `(.venv)`).
* **PyTorch fails to install**: use Python 3.12 or 3.11 (very new Python versions may not have PyTorch builds yet).
* **Slow or out of memory**: the demo needs about 3 GB of free RAM. Close other programs (browsers), or use `--fast`.

## Reproducing everything (not for the live demo)

The full experiments take several hours on a laptop:

```
python run_kids0.py --part main --seeds 5
python run_kids0.py --part ablation --seeds 3
python run_v2.py --part main --seeds 5
python run_v2.py --part capture --seeds 3
python run_v2.py --part ablation --seeds 3
python run_v2.py --part extra --seeds 3
```
