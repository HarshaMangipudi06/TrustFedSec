# TrustFed-Sec

A local CPU-only research demonstration of federated binary intrusion detection, simulated poisoning, and validation-assisted trust-weighted aggregation. It is not a production security product. The default dataset is **synthetic** and must never be reported as UNSW-NB15.

## Windows setup

From PowerShell in this directory, with Python 3.10–3.13 recommended:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The browser opens at the URL shown by Streamlit. Run tests with `.\.venv\Scripts\python.exe -m pytest -q`. Internet access is needed to install packages, not to run the synthetic demo. Results are saved as JSON in `results/`; reset clears the dashboard session, not files already saved.

## Dataset

Select **Synthetic demonstration** to run without files. To use UNSW-NB15, select **Local UNSW-NB15 CSV** and upload one or more compatible CSVs with headers and a binary `label` column (0 = normal, 1 = attack). Files are concatenated and split anew, including if official pre-split files are uploaded. The application labels uploaded data as user-supplied; it cannot verify provenance. Do not upload raw, headerless files without assigning correct column names first. The loader drops `id`, `attack_cat`, common IP and port identifiers, and the label; it imputes missing numeric values and standardizes them, one-hot encodes categorical values with unknown-category handling, and fits preprocessing only on the training split. Review other dataset-specific leakage fields before using different CSVs. Memory use rises with large files and categorical cardinality.

A stratified 70/15/15 split creates training, trusted validation and held-out test sets. IID partitions shuffle training records; non-IID uses class-skewed Dirichlet allocation. Validation is used for trust and progress metrics, so final comparisons use the untouched test set. Very small or severely imbalanced datasets may fail stratification. Client sample counts in this simulation are known locally and capped at 100,000 for aggregation.

## Run the demonstration

1. Choose a dataset, partition, seed, attack, client count, compromised count and rounds. The default five-client sign-flip setup is a useful starting point.
2. Run **Normal FedAvg**, **Poisoned FedAvg**, and **TrustFed-Sec** with unchanged controls. Results from different settings are hidden rather than compared. In the two attacked runs, the same first N clients are compromised; label flips use deterministic per-client/per-round seeds. As models diverge, later updates naturally differ.
3. Inspect held-out test metrics, validation trajectories and client details. Select a round and client to see the explanation. Review flags are statistical signals, not proof of compromise. The app reports actual results even if the defense does not improve them.

## Method

Each client receives the same parameters and performs full-batch gradient descent on binary logistic cross-entropy. FedAvg weights client parameter deltas by their sample counts. The coordinator adds the weighted mean delta to the current model, equivalent to averaging client parameters.

Four attacks are available. Sign flipping multiplies the update by a negative multiplier. Label flipping changes a selected fraction of local labels before training. Constrain-and-scale trains on flipped labels and submits a magnified update. Cosine adaptive skips local training: it takes the median of the honest updates in that round, adds a small component that increases validation loss, then rescales to the honest median's norm. That last attacker sees both the honest updates and the validation set, so it is deliberately stronger than a realistic one.

For each round, the norm signal is `abs(log((norm_i + eps)/(median_norm + eps)))`; direction is cosine similarity to the coordinate-wise median update, with zero vectors handled explicitly. Validation impact is candidate loss minus current global validation loss. The bounded quality components are:

- norm: `exp(-norm_deviation / norm_threshold)`;
- direction: `clip((cosine - cosine_threshold)/(1 - cosine_threshold), 0, 1)`;
- validation: `exp(-max(0, loss_delta)/validation_threshold)`.

Evidence is `0.25*norm + 0.25*direction + 0.50*validation`; trust is `clip(beta*previous_trust + (1-beta)*evidence, trust_floor, 1)`, initially 1. The aggregation weight is `trust_i * min(n_i, cap)` normalized across clients. A floor softly retains all clients; if weights are invalid or zero, the coordinator falls back to equal weights. Thresholds, beta and floor are adjustable. Indicators are computed for all scenarios, but FedAvg does not use them. The validation component is bounded and cannot dominate by magnitude. A review status flags any threshold crossing and is separate from the trust calculation.

Arm D adds an FPR budget of 0.02 by default. For each client the coordinator computes the validation FPR of `theta + delta_i`, then keeps the largest fraction `alpha_i in [0, 1]` for which that FPR stays within `current_FPR + budget`. It renormalises the surviving weights and, if the combined update still breaches the cap, shrinks the combined step by the same rule. The budget is a limit on the rise in one round, not an absolute FPR target, so an all-zero update is always feasible. This is not a calibrated guarantee: the same validation set is reused every round, an attacker with access to it can evade the cap, and the budget trades attack detection against false alarms.

## Limitations and future work

Central trusted validation may be unrepresentative, costly or compromised; repeated selection against it risks overfitting. Legitimate non-IID updates may be down-weighted; the median can be manipulated by collusion, and adaptive attackers can evade these signals. The floor does not guarantee robustness; no method guarantees protection. Simulation runs centrally without privacy, secure aggregation, real devices, network transport, adversarial identity verification or verified sample counts. There is no differential privacy or production monitoring. Dense one-hot matrices can consume substantial RAM. The communication estimate excludes transport overhead. Future work: stronger defenses and baselines, validated dataset-specific leakage audits, streaming preprocessing, independent benchmarks across seeds and attack budgets, secure aggregation-compatible approaches, and deployment/security review.

## Problem and proposed solution

A malicious participant can send a manipulated update that degrades a shared intrusion detector. TrustFed-Sec examines update magnitude, direction and performance on a trusted validation set, then combines the evidence with historical trust to softly reduce some contributions. This is a CPU-based federated learning simulation, not live network intrusion monitoring. Real UNSW-NB15 validation has not yet been established. See [ARCHITECTURE.md](ARCHITECTURE.md) for the workflow, [DEMO.md](DEMO.md) for the demonstration, and [JUDGE_QA.md](JUDGE_QA.md) for technical questions.

## Technology and project structure

Python, NumPy, pandas, scikit-learn, Streamlit, Plotly and pytest. The entry point is `app.py`. The core modules are `src/dataset.py` (splits and preprocessing), `model.py` (local training), `attacks.py`, `trust.py` (indicators, scoring and aggregation), `coordinator.py` (round orchestration), and `evaluation.py`. `config.py` contains defaults, `tests/` contains checks, `data/` is for local datasets, `results/` contains experiment records, and `docs/screenshots/` is reserved for real screenshots.

## Saved synthetic demonstration results

These values are supported by the three checked-in JSON records in `results/`; they are **synthetic demonstration data**, not UNSW-NB15 benchmark results. The records specify seed 42, 6,000 samples, 18 features, five IID clients, two compromised clients, sign-flip multiplier 5, and ten rounds. Values below are rounded to four decimals. The JSON records do not establish dataset provenance independently or constitute an external benchmark.

| Held-out test metric | Normal FedAvg | Poisoned FedAvg | TrustFed-Sec |
|---|---:|---:|---:|
| Accuracy | 0.7700 | 0.2878 | 0.7433 |
| Precision | 0.7104 | 0.2075 | 0.6619 |
| Recall | 0.6754 | 0.3043 | 0.6754 |
| Macro F1 | 0.7544 | 0.2857 | 0.7296 |
| False positive rate | 0.1712 | 0.7225 | 0.2144 |
| Attack recall | 0.6754 | 0.3043 | 0.6754 |

In the final TrustFed-Sec round, the saved weights for clients 1–5 are 0.0316, 0.0316, 0.3134, 0.3100 and 0.3133 respectively (rounded). Clients 1 and 2 are the simulated compromised clients. The comparison describes one run only; the method does not guarantee similar outcomes on other data or attacks. The files contain no raw training data, but do include model parameters and experiment details. Newly generated JSON records are ignored by Git to avoid accidental publication.

## Multi-seed benchmark

A single run is a demo; the benchmark is the evidence. Run the same four arms across independent seeds:

```bash
python3 scripts/bench.py --seeds 30 --rounds 10
```

It writes `results/benchmark_summary.json`. Each seed gets its own split, partition and starting model; all four arms use that same setup. Arm D is arm C with the FPR guard switched on. Contrasts are paired by seed and reported with Wilcoxon signed-rank (primary), paired t (secondary), Cohen's d_z, a 95% CI on the paired difference, and Holm correction across the five contrasts. Test-row bootstrap CIs (B=1000 by default) are stored per run. The data is the synthetic demonstration unless `--csv path/to/file.csv ...` is supplied. See `tests/test_bench.py` and `tests/test_fpr.py` for the checks.

Checked-in result, 30 seeds × 10 rounds, synthetic sign flip (2 of 5 clients, multiplier 5):

| Arm | Accuracy | FPR | Macro F1 |
|---|---:|---:|---:|
| A clean | 0.8094 ± 0.0352 | 0.1359 ± 0.0343 | 0.7954 |
| B poisoned | 0.2615 ± 0.0352 | 0.7457 ± 0.0491 | 0.2580 |
| C TrustFed-Sec | 0.7761 ± 0.0383 | 0.1789 ± 0.0427 | 0.7619 |
| D TrustFed-Sec + FPR guard | 0.7977 ± 0.0342 | 0.1526 ± 0.0388 | 0.7837 |

Paired by seed: C − B is +0.5146 accuracy (95% CI 0.4897 to 0.5395, d_z 7.70, Holm p = 6.9e-6); D − C is +0.0216 accuracy (CI 0.0171 to 0.0261, d_z 1.79) and −0.0263 FPR (CI −0.0332 to −0.0194). The guard narrows the remaining gap: D − A is −0.0118 accuracy (CI −0.0153 to −0.0083, d_z −1.26) and +0.0167 FPR (CI 0.0100 to 0.0235). It still does not match clean training, and it raises false alarms. These are synthetic results, not an UNSW-NB15 benchmark.

### Other attacks

Two further 30-seed sweeps use the same four arms (`results/bench_constrain.json`, `results/bench_adaptive.json`). Both are negative results:

- Constrain-and-scale: clean 0.8094 accuracy, poisoned 0.8071, TrustFed-Sec 0.8106. B − A = −0.0023 (Holm p = 1.0). The attack barely moves this model, so there is nothing for the defense to recover.
- Cosine adaptive: poisoned 0.8071, TrustFed-Sec 0.8071, D − C = 0.0000 (Holm p = 1.0). With a 5% harmful component and full knowledge of the honest updates and validation set, the crafted update is effectively invisible — and also effectively harmless. A stronger adaptive attacker is still needed before claiming robustness against one.

### UNSW-NB15 sample

A 30-seed sweep on 12,000 rows sampled from third-party mirror CSVs (`results/bench_unsw.json`; see [docs/DATA.md](docs/DATA.md)). The same sign-flip attack, four arms, 10 rounds:

| Arm | Accuracy | FPR | Macro F1 |
|---|---:|---:|---:|
| A clean | 0.8201 ± 0.0127 | 0.3320 ± 0.0197 | 0.7967 |
| B poisoned | 0.2661 ± 0.0129 | 0.8742 ± 0.0214 | 0.2427 |
| C TrustFed-Sec | 0.7933 ± 0.0151 | 0.2874 ± 0.0410 | 0.7754 |
| D + FPR guard | 0.7573 ± 0.0161 | 0.1777 ± 0.0506 | 0.7501 |

C − B = +0.5272 accuracy (CI 0.5190 to 0.5355, d_z 23.89, Holm p = 8.6e-6). D − C = −0.0360 accuracy and −0.1097 FPR, both significant. On real traffic the guard buys a large FPR reduction at a measurable accuracy cost, and the baseline FPR is already 0.33. This is a sampled, mirrored dataset with no verified provenance — not a benchmark claim.

## Screenshots

Two real captures from the default synthetic run are in `docs/screenshots/`: the three-arm comparison and the TrustFed-Sec client inspection at round 10. Measured values are in `results/`. See [docs/screenshots/README.md](docs/screenshots/README.md) for the capture settings.

## Team contributions

- Team member / role: **[add name and contribution]**
- Team member / role: **[add name and contribution]**

No license has been selected; obtain an appropriate license decision before treating the repository as licensed for reuse.
