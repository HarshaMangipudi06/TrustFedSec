# Feature status

Implemented: synthetic and uploaded CSV data; train-only preprocessing; IID and class-skewed client partitions; logistic local training; ordinary FedAvg; sign flip, label flip, constrain-and-scale and cosine-adaptive attacks; bounded update, validation and historical trust signals; trust-weighted aggregation with floor and fallback; an optional FPR-budgeted guard that shrinks updates which would raise validation FPR above a per-round cap; per-round dashboard, held-out metrics, reproducible seeds, JSON output, and an automated four-arm multi-seed benchmark with seed-paired statistics.

Planned: real distributed clients, secure aggregation, privacy guarantees, attacker-without-validation-secrets evaluation, large-scale dataset streaming, verified client sample counts, class-wise FPR budgets, and production intrusion monitoring.
