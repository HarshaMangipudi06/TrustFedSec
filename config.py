from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Config:
    seed: int = 42
    clients: int = 5
    compromised: int = 2
    attack: str = 'sign flip'
    strength: float = 5.0
    poison_fraction: float = 0.5
    adaptive_fraction: float = 0.05
    rounds: int = 10
    partition: str = 'IID'
    epochs: int = 3
    learning_rate: float = 0.15
    beta: float = 0.6
    norm_threshold: float = 1.0
    cosine_threshold: float = 0.2
    validation_threshold: float = 0.02
    trust_floor: float = 0.10
    fpr_budget: float = 0.02
    max_samples: int = 100000
    max_rows: int | None = None
