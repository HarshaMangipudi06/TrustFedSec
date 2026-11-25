import numpy as np

from src.model import predict_proba


def flip_labels(y, fraction, rng):
    result = y.copy(); n = int(round(len(y) * np.clip(fraction, 0, 1)))
    if n: idx = rng.choice(len(y), n, replace=False); result[idx] = 1-result[idx]
    return result


def manipulate(update, strength):
    return -float(strength) * update


def constrain_and_scale(update, strength):
    """Train on poisoned labels, then submit a magnified version of that update."""
    return float(strength) * update


def harm_direction(theta, X, y):
    """Gradient that INCREASES validation loss, used as the adaptive attacker's target."""
    residual = (predict_proba(np.asarray(X), np.asarray(theta)) - np.asarray(y)) / max(1, len(y))
    return np.concatenate([np.asarray(X).T @ residual, np.array([residual.sum()])])


def cosine_adaptive(reference, harm, fraction=0.05):
    """Update aligned with the median honest update plus a small harmful component.

    Keeps the direction cosine near 1 while quietly pushing the global model
    toward higher validation loss. The attacker here knows the honest updates
    and the validation set; that is a strong, deliberately unfair adversary.
    """
    ref = np.asarray(reference, dtype=float)
    ref_norm = np.linalg.norm(ref)
    if ref_norm <= 1e-12:
        return np.zeros_like(ref)
    harm_norm = np.linalg.norm(harm)
    if harm_norm <= 1e-12:
        return ref.copy()
    fraction = float(np.clip(fraction, 0, 1))
    u_ref = ref / ref_norm
    u_harm = np.asarray(harm, dtype=float) / harm_norm
    # Subtract the component already pointing along ref so the harmful part
    # does not inflate the update norm.
    u_harm = u_harm - (u_harm @ u_ref) * u_ref
    hn = np.linalg.norm(u_harm)
    if hn <= 1e-12:
        return ref.copy()
    crafted = (1 - fraction) * u_ref + fraction * u_harm / hn
    return ref_norm * crafted / np.linalg.norm(crafted)
