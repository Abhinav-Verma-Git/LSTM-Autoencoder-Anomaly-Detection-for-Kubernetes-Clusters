"""Flow-level relabelling of the Sever & Dogan (2023) attack captures.

The published dataset labels EVERY flow of an attack capture with the scenario id, including ordinary
cluster control-plane traffic recorded during the attack (OVN southbound DB :6642, DNS :53, health
probes :8080/:8181/:7472, node-to-API-server :6443, Grafana update checks to GitHub/grafana.com).

The attacker's own address is not visible: Kube-OVN source-NATs all external traffic to the join-subnet
gateway 100.64.0.2. We therefore identify attack flows by the attacked endpoint and, where the dataset
paper documents a compromise, by the compromised pod's connections:

  R1  inbound request from outside the cluster (Src IP = gateway 100.64.0.2) to the scenario's target
      application port: Grafana :3000 (scenarios 1, 5), Node-RED :1880 (2, 3, 4), InfluxDB :8086 (6).
  R2  scenarios 3 and 4 only (Node-RED RCE / container escape): every flow to or from the compromised
      Node-RED pod, identified as the destination of the R1 flows in that capture (reverse shell to
      :9001, kubectl/tool downloads over :443, Kubernetes API calls with the stolen token on :6443).
      In all benign captures this pod only ever receives inbound :1880 requests.

Scenarios 7-10 (runc x2, containerd, kubelet CVEs) act on the host; none of their flows matches R1/R2
and every (src, dst, dport) triple in those captures also occurs in benign traffic, so all their flows
are relabelled benign. All flows not matched by R1/R2 are relabelled benign (0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GATEWAY = "100.64.0.2"
TARGET_PORT = {1: 3000, 5: 3000, 2: 1880, 3: 1880, 4: 1880, 6: 8086}
COMPROMISED_POD_SCENARIOS = {3, 4}


def flow_labels(frame: pd.DataFrame, scenario: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (label per flow: 0 or scenario id, rule per flow: '', 'R1' or 'R2')."""
    n = len(frame)
    rule = np.array([""] * n, dtype=object)
    if scenario == 0 or scenario not in TARGET_PORT:
        return np.zeros(n, dtype=int), rule
    src, dst, dport = frame["Src IP"].to_numpy(), frame["Dst IP"].to_numpy(), frame["Dst Port"].to_numpy()
    r1 = (src == GATEWAY) & (dport == TARGET_PORT[scenario])
    rule[r1] = "R1"
    if scenario in COMPROMISED_POD_SCENARIOS:
        victims = set(dst[r1])
        r2 = (np.isin(src, list(victims)) | np.isin(dst, list(victims))) & ~r1
        rule[r2] = "R2"
    lab = np.where(rule != "", scenario, 0)
    return lab.astype(int), rule


if __name__ == "__main__":
    import json
    import sys
    from data import SCENARIO_NAMES, load_captures

    caps = load_captures(sys.argv[1] if len(sys.argv) > 1 else "data_k8s")
    rows = []
    for c in caps:
        if c.label == 0:
            continue
        lab, rule = flow_labels(c.frame, c.label)
        rows.append({"scenario": c.label, "name": SCENARIO_NAMES[c.label], "flows": len(c.frame),
                     "attack_R1": int((rule == "R1").sum()), "attack_R2": int((rule == "R2").sum()),
                     "attack": int((lab > 0).sum()), "relabelled_benign": int((lab == 0).sum())})
    df = pd.DataFrame(rows).sort_values("scenario")
    print(df.to_string(index=False))
    print("TOTAL flows", df.flows.sum(), "attack", df.attack.sum(), "relabelled benign", df.relabelled_benign.sum())
    json.dump(rows, open("results_relabel/relabel_counts.json", "w"), indent=1)
