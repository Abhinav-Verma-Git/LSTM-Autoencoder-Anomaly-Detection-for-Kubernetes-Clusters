
## Protocol A
```
{"labels": "flow", "attack_split": "blocked", "window": 10, "log1p": false, "n_raw_features": 80, "n_features_used": 74, "n_train_windows": 36281, "val": {"normal": 13087, "attack": 1113, "normal_from_attack_captures": 993, "attack_by_scenario": {"1": 2, "2": 1092, "3": 4, "4": 8, "6": 7}}, "test": {"normal": 12984, "attack": 1134, "normal_from_attack_captures": 888, "attack_by_scenario": {"2": 1114, "3": 6, "4": 3, "5": 1, "6": 10}}}
```
| Model | ROC-AUC | PR-AUC | exploits AUC | recon AUC | any-flow-label AUC |
|---|---|---|---|---|---|
| LSTM-AE | 0.431 ± 0.005 | 0.113 ± 0.005 | 0.819 ± 0.009 | 0.424 ± 0.005 | 0.495 ± 0.005 |
| Dense-AE | 0.431 ± 0.003 | 0.092 ± 0.001 | 0.808 ± 0.003 | 0.424 ± 0.003 | 0.493 ± 0.002 |
| IForest | 0.417 ± 0.017 | 0.099 ± 0.006 | 0.810 ± 0.005 | 0.410 ± 0.017 | 0.476 ± 0.014 |
| PCA | 0.453 | 0.084 | 0.798 | 0.447 | 0.509 |

f1
| Model | P | R | F1 | FPR | FPR bg-in-attack-caps | FPR benign caps | R exploits | R recon |
|---|---|---|---|---|---|---|---|---|
| LSTM-AE | 0.171 ± 0.058 | 0.320 ± 0.372 | 0.166 ± 0.012 | 0.240 ± 0.408 | 0.346 ± 0.366 | 0.233 ± 0.411 | 0.420 ± 0.325 | 0.319 ± 0.372 |
| Dense-AE | 0.080 ± 0.000 | 0.996 ± 0.002 | 0.149 ± 0.000 | 0.995 ± 0.003 | 1.000 | 0.995 ± 0.003 | 1.000 | 0.996 ± 0.002 |
| IForest | 0.088 ± 0.004 | 0.952 ± 0.008 | 0.162 ± 0.007 | 0.859 ± 0.044 | 1.000 | 0.848 ± 0.047 | 1.000 | 0.951 ± 0.008 |
| PCA | 0.117 | 0.293 | 0.168 | 0.192 | 0.401 | 0.177 | 0.450 | 0.290 |

p99
| Model | P | R | F1 | FPR | FPR bg-in-attack-caps | FPR benign caps | R exploits | R recon |
|---|---|---|---|---|---|---|---|---|
| LSTM-AE | 0.405 ± 0.016 | 0.072 ± 0.006 | 0.122 ± 0.009 | 0.009 ± 0.000 | 0.083 ± 0.005 | 0.004 ± 0.000 | 0.200 | 0.069 ± 0.006 |
| Dense-AE | 0.268 ± 0.012 | 0.042 ± 0.003 | 0.072 ± 0.004 | 0.010 ± 0.000 | 0.076 ± 0.001 | 0.005 ± 0.000 | 0.200 | 0.039 ± 0.003 |
| IForest | 0.380 ± 0.029 | 0.033 ± 0.003 | 0.060 ± 0.006 | 0.005 ± 0.000 | 0.019 ± 0.002 | 0.004 ± 0.000 | 0.000 | 0.033 ± 0.003 |
| PCA | 0.095 | 0.049 | 0.064 | 0.040 | 0.070 | 0.038 | 0.200 | 0.046 |

per scenario (LSTM-AE): {"2": [1114, 0.424, 0.319], "3": [6, 0.904, 0.533], "4": [3, 0.849, 0.467], "5": [1, 0.913, 0.2], "6": [10, 0.75, 0.36]}

## Protocol B
```
{"labels": "flow", "attack_split": "chrono", "window": 10, "log1p": false, "n_raw_features": 80, "n_features_used": 74, "n_train_windows": 36281, "val": {"normal": 13109, "attack": 1047, "normal_from_attack_captures": 1015, "attack_by_scenario": {"1": 2, "2": 1024, "3": 6, "4": 6, "6": 9}}, "test": {"normal": 12962, "attack": 1200, "normal_from_attack_captures": 866, "attack_by_scenario": {"2": 1182, "3": 4, "4": 5, "5": 1, "6": 8}}}
```
| Model | ROC-AUC | PR-AUC | exploits AUC | recon AUC | any-flow-label AUC |
|---|---|---|---|---|---|
| LSTM-AE | 0.284 ± 0.012 | 0.064 ± 0.001 | 0.830 ± 0.007 | 0.276 ± 0.012 | 0.344 ± 0.011 |
| Dense-AE | 0.297 ± 0.003 | 0.061 ± 0.000 | 0.814 ± 0.006 | 0.289 ± 0.003 | 0.354 ± 0.003 |
| IForest | 0.300 ± 0.024 | 0.072 ± 0.002 | 0.760 ± 0.010 | 0.293 ± 0.025 | 0.351 ± 0.022 |
| PCA | 0.316 | 0.061 | 0.791 | 0.308 | 0.369 |

f1
| Model | P | R | F1 | FPR | FPR bg-in-attack-caps | FPR benign caps | R exploits | R recon |
|---|---|---|---|---|---|---|---|---|
| LSTM-AE | 0.099 ± 0.029 | 0.042 ± 0.002 | 0.058 ± 0.004 | 0.039 ± 0.012 | 0.084 ± 0.018 | 0.035 ± 0.012 | 0.289 ± 0.114 | 0.039 ± 0.000 |
| Dense-AE | 0.044 ± 0.001 | 0.046 ± 0.004 | 0.045 ± 0.002 | 0.093 ± 0.006 | 0.085 ± 0.009 | 0.094 ± 0.006 | 0.344 ± 0.072 | 0.042 ± 0.003 |
| IForest | 0.036 ± 0.005 | 0.033 ± 0.001 | 0.034 ± 0.002 | 0.084 ± 0.015 | 0.123 ± 0.021 | 0.082 ± 0.015 | 0.067 ± 0.025 | 0.033 ± 0.001 |
| PCA | 0.061 | 0.130 | 0.083 | 0.186 | 0.313 | 0.177 | 0.444 | 0.125 |

p99
| Model | P | R | F1 | FPR | FPR bg-in-attack-caps | FPR benign caps | R exploits | R recon |
|---|---|---|---|---|---|---|---|---|
| LSTM-AE | 0.292 ± 0.031 | 0.019 ± 0.005 | 0.035 ± 0.008 | 0.004 ± 0.000 | 0.033 ± 0.001 | 0.002 ± 0.000 | 0.167 | 0.017 ± 0.005 |
| Dense-AE | 0.167 ± 0.048 | 0.011 ± 0.004 | 0.020 ± 0.007 | 0.005 ± 0.000 | 0.031 ± 0.000 | 0.003 ± 0.000 | 0.167 | 0.008 ± 0.004 |
| IForest | 0.321 ± 0.030 | 0.019 ± 0.001 | 0.036 ± 0.002 | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.003 ± 0.000 | 0.000 | 0.019 ± 0.001 |
| PCA | 0.066 | 0.023 | 0.034 | 0.031 | 0.031 | 0.031 | 0.167 | 0.021 |

per scenario (LSTM-AE): {"2": [1182, 0.276, 0.039], "3": [4, 0.85, 0.25], "4": [5, 0.914, 0.44], "5": [1, 0.907, 0.0], "6": [8, 0.759, 0.25]}

## capture_labels
| Config | ROC-AUC | PR-AUC | exploits | recon | F1 | epochs |
|---|---|---|---|---|---|---|
| LSTM-AE(capture-labels) | 0.595 ± 0.001 | 0.284 ± 0.002 | 0.779 ± 0.004 | 0.478 ± 0.004 | 0.296 ± 0.002 | 60 |

## ablation
| Config | ROC-AUC | PR-AUC | exploits | recon | F1 | epochs |
|---|---|---|---|---|---|---|
| LSTM-AE(T=1) | 0.685 ± 0.055 | 0.142 ± 0.025 | 0.888 ± 0.001 | 0.681 ± 0.056 | 0.236 ± 0.043 | 60 |
| LSTM-AE(T=5) | 0.513 ± 0.017 | 0.104 ± 0.004 | 0.842 ± 0.002 | 0.507 ± 0.017 | 0.158 ± 0.001 | 54 |
| LSTM-AE(T=20) | 0.438 ± 0.004 | 0.124 ± 0.007 | 0.839 ± 0.002 | 0.431 ± 0.004 | 0.211 ± 0.010 | 33 |
| LSTM-AE(log1p) | 0.422 ± 0.016 | 0.068 ± 0.004 | 0.854 ± 0.001 | 0.415 ± 0.016 | 0.161 ± 0.002 | 60 |

## extra
| Config | ROC-AUC | PR-AUC | exploits | recon | F1 | epochs |
|---|---|---|---|---|---|---|
| LSTM-AE(200ep,mean) | 0.450 ± 0.011 | 0.119 ± 0.002 | 0.821 ± 0.005 | 0.444 ± 0.011 | 0.178 ± 0.014 | 97 |
| LSTM-AE(200ep,max) | 0.441 ± 0.013 | 0.099 ± 0.001 | 0.797 ± 0.013 | 0.435 ± 0.014 | 0.157 ± 0.002 | 97 |
| LSTM-AE(200ep,last) | 0.473 ± 0.007 | 0.115 ± 0.001 | 0.869 ± 0.029 | 0.466 ± 0.008 | 0.151 ± 0.003 | 97 |

controls seed0 A: {"LSTM-AE": {"exploit_vs_benign_captures": 0.8331225198412698, "exploit_vs_capture_background": 0.5953828828828829, "capture_background_vs_benign": 0.780684143965394, "n_exploit": 20, "n_bg": 888, "n_benign": 12096}, "Dense-AE": {"exploit_vs_benign_captures": 0.8239996693121694, "exploit_vs_capture_background": 0.6095720720720721, "capture_background_vs_benign": 0.7578075657502741, "n_exploit": 20, "n_bg": 888, "n_benign": 12096}, "IForest": {"exploit_vs_benign_captures": 0.8312582671957671, "exploit_vs_capture_background": 0.5847972972972972, "capture_background_vs_benign": 0.7899801773499691, "n_exploit": 20, "n_bg": 888, "n_benign": 12096}, "PCA": {"exploit_vs_benign_captures": 0.8110532407407407, "exploit_vs_capture_background": 0.6188626126126127, "capture_background_vs_benign": 0.7390982872753706, "n_exploit": 20, "n_bg": 888, "n_benign": 12096}}

controls seed0 B: {"LSTM-AE": {"exploit_vs_benign_captures": 0.8399746472663139, "exploit_vs_capture_background": 0.6607005388760585, "capture_background_vs_benign": 0.7532137530243044, "n_exploit": 18, "n_bg": 866, "n_benign": 12096}, "Dense-AE": {"exploit_vs_benign_captures": 0.8238122795414461, "exploit_vs_capture_background": 0.6778932512188863, "capture_background_vs_benign": 0.7313367578234784, "n_exploit": 18, "n_bg": 866, "n_benign": 12096}, "IForest": {"exploit_vs_benign_captures": 0.7820859053497942, "exploit_vs_capture_background": 0.5211059789581729, "capture_background_vs_benign": 0.7711540928919682, "n_exploit": 18, "n_bg": 866, "n_benign": 12096}, "PCA": {"exploit_vs_benign_captures": 0.7985927395649618, "exploit_vs_capture_background": 0.68289710033359, "capture_background_vs_benign": 0.7011535697484024, "n_exploit": 18, "n_bg": 866, "n_benign": 12096}}

## matched
{
 "prevalence": 0.6154877205119336,
 "trivial_f1": 0.7619837807456574,
 "A/LSTM-AE": {
  "recall": 0.35361552028218696,
  "precision": 0.5071823094558836,
  "f1": 0.41670071656834107,
  "accuracy": 0.3906765362857546
 },
 "A/Dense-AE": {
  "recall": 0.3800705467372134,
  "precision": 0.5251981041254847,
  "f1": 0.4410013102521454,
  "accuracy": 0.40695928021464167
 },
 "A/IForest": {
  "recall": 0.3403880070546737,
  "precision": 0.49765180632550365,
  "f1": 0.4042641026302902,
  "accuracy": 0.38253516432131096
 },
 "A/PCA": {
  "recall": 0.43209876543209874,
  "precision": 0.5570443426750974,
  "f1": 0.48668017961817395,
  "accuracy": 0.43898200994145303
 },
 "B/LSTM-AE": {
  "recall": 0.1575,
  "precision": 0.31430834500125293,
  "f1": 0.20984607356856258,
  "accuracy": 0.26996984175025945
 },
 "B/Dense-AE": {
  "recall": 0.19,
  "precision": 0.3560717765080702,
  "f1": 0.24778294886124186,
  "accuracy": 0.28997319266689725
 },
 "B/IForest": {
  "recall": 0.15,
  "precision": 0.303889616304224,
  "f1": 0.2008569520351435,
  "accuracy": 0.26535368384641994
 },
 "B/PCA": {
  "recall": 0.25333333333333335,
  "precision": 0.42439113370639076,
  "f1": 0.3172747208273672,
  "accuracy": 0.32895408163265305
 }
}