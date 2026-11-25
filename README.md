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

Each client receives the same parameters and performs full-batch gradient descent on binary logistic cross-entropy. FedAvg weights client parameter deltas by their sample counts. Sign flipping multiplies an update by the negative attack multiplier; label flipping changes a selected fraction of local labels before training. The coordinator adds the weighted mean delta to the current model, equivalent to averaging client parameters.

For each round, the norm signal is `abs(log((norm_i + eps)/(median_norm + eps)))`; direction is cosine similarity to the coordinate-wise median update, with zero vectors handled explicitly. Validation impact is candidate loss minus current global validation loss. The bounded quality components are:

- norm: `exp(-norm_deviation / norm_threshold)`;
- direction: `clip((cosine - cosine_threshold)/(1 - cosine_threshold), 0, 1)`;
- validation: `exp(-max(0, loss_delta)/validation_threshold)`.

Evidence is `0.25*norm + 0.25*direction + 0.50*validation`; trust is `clip(beta*previous_trust + (1-beta)*evidence, trust_floor, 1)`, initially 1. The aggregation weight is `trust_i * min(n_i, cap)` normalized across clients. A floor softly retains all clients; if weights are invalid or zero, the coordinator falls back to equal weights. Thresholds, beta and floor are adjustable. Indicators are computed for all scenarios, but FedAvg does not use them. The validation component is bounded and cannot dominate by magnitude. A review status flags any threshold crossing and is separate from the trust calculation.

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

## Screenshots

Two real captures from the default synthetic run are in `docs/screenshots/`: the three-arm comparison and the TrustFed-Sec client inspection at round 10. Measured values are in `results/`. See [docs/screenshots/README.md](docs/screenshots/README.md) for the capture settings.

## Team contributions

- Team member / role: **[add name and contribution]**
- Team member / role: **[add name and contribution]**

No license has been selected; obtain an appropriate license decision before treating the repository as licensed for reuse.
