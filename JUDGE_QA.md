# Judge questions

1. **Why logistic regression?** Shared, interpretable parameter vectors make exact aggregation and CPU demonstrations straightforward; capacity is limited.
2. **What makes comparison fair?** Matching seed, split, client partitions, initial model, rounds and attack selection; the held-out test is shared. Later local updates differ as global models diverge.
3. **Can validation leak into training?** The preprocessing fit and client training use only training records; validation guides trust and is reused, so it is not an unbiased final evaluation. Test is separate.
4. **What if a benign client is non-IID?** Its update may appear anomalous and be down-weighted; smoothing and a trust floor limit but do not remove this risk.
5. **Why use the median direction?** It provides a simple robust reference, but a coordinated majority can move it and heterogeneous populations can make it misleading.
6. **How are attacks simulated?** Sign flipping scales a negative local update; label flipping changes a random fraction of compromised clients’ training labels; constrain-and-scale trains on flipped labels then magnifies the update; cosine adaptive crafts an update aligned with the median honest update plus a small harmful component. None is real malware, and the adaptive attacker is given the validation set, which makes it stronger than a realistic one.
7. **What does trust mean?** A bounded heuristic combining norm, direction, validation loss and history; it is not a calibrated probability of honesty.
8. **Can attackers evade it?** Yes. The cosine-adaptive arm mimics benign direction and norm, and the benchmark measures what happens. No guarantee is claimed. An attacker without access to the validation set is not yet evaluated.
9. **What is communication overhead?** An estimate of full float64 parameter vectors sent to and returned from each client per round, excluding protocol costs.
10. **Is this privacy-preserving?** No. The central simulation sees updates and holds trusted validation data; there is no secure aggregation or differential privacy.
11. **Have you tried real data?** Yes, a 30-seed run on 12,000 sampled rows from a third-party mirror of UNSW-NB15. The defense recovers 0.527 accuracy over poisoned FedAvg; the FPR guard cuts FPR by 0.110 but costs 0.036 accuracy. Provenance is not independently verified, so this is a sample result, not a benchmark claim. See `docs/DATA.md` and `results/bench_unsw.json`.
