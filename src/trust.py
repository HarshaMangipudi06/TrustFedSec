import numpy as np

from src.model import predict_proba


def weights_for(updates, counts, trust=None, max_samples=100000, floor=0.1):
    counts = np.clip(np.asarray(counts, dtype=float), 0, max_samples)
    if trust is None:
        weights = counts
    else:
        # Invalid trust is not evidence of reliability; apply the minimum trust.
        safe_trust = np.nan_to_num(np.asarray(trust, dtype=float), nan=floor, posinf=floor, neginf=floor)
        weights = counts * np.clip(safe_trust, floor, 1.0)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(updates))
    return weights / weights.sum()


def aggregate(updates, counts, trust=None, max_samples=100000, floor=0.1):
    weights = weights_for(updates, counts, trust, max_samples, floor)
    return np.average(np.asarray(updates), axis=0, weights=weights), weights


def indicators(updates):
    updates = np.asarray(updates); norms = np.linalg.norm(updates, axis=1)
    reference = np.median(updates, axis=0); refnorm = np.linalg.norm(reference)
    cos = np.zeros(len(updates))
    for i, u in enumerate(updates):
        if norms[i] > 1e-12 and refnorm > 1e-12: cos[i] = np.clip(np.dot(u, reference)/(norms[i]*refnorm), -1, 1)
        elif norms[i] <= 1e-12 and refnorm <= 1e-12: cos[i] = 1.0
    median = float(np.median(norms)); deviation = np.abs(np.log((norms+1e-12)/(median+1e-12)))
    return norms, cos, deviation


def score(previous, norm_deviation, cosine, loss_delta, cfg):
    norm_quality = np.exp(-norm_deviation / max(cfg.norm_threshold, 1e-9))
    direction_quality = np.clip((cosine-cfg.cosine_threshold)/(1-cfg.cosine_threshold), 0, 1)
    validation_quality = np.exp(-max(0, loss_delta)/max(cfg.validation_threshold, 1e-9))
    evidence = 0.25*norm_quality + 0.25*direction_quality + 0.50*validation_quality
    return float(np.clip(cfg.beta*previous+(1-cfg.beta)*evidence, cfg.trust_floor, 1)), float(evidence)


def validation_fpr(theta, X_val, y_val):
    """False positive rate of a candidate parameter vector on the trusted validation set."""
    pred = (predict_proba(np.asarray(X_val), np.asarray(theta)) >= 0.5).astype(int)
    neg = np.asarray(y_val) == 0
    if not neg.any():
        return 1.0
    fp = int(np.sum((pred == 1) & neg))
    tn = int(np.sum((pred == 0) & neg))
    return fp / (fp + tn) if fp + tn else 1.0


def fpr_budget_weights(theta, weights, updates, X_val, y_val, budget=0.02, steps=10):
    """Shrink updates whose own candidate would raise validation FPR too much.

    The budget is an absolute increase over the FPR of the current global model,
    not an absolute FPR target: a 5% cap means 'no more than +0.05 this round'.
    Each client's weight is scaled by the largest alpha in [0, 1] that satisfies
    FPR(theta + alpha * delta_i) <= current_fpr + budget. Weights are then
    renormalised and one second pass caps the FPR of the combined update.

    Returns (weights, diagnostics) where diagnostics['scale'] multiplies the
    aggregate delta; alpha < 1 means that update was partly or fully dropped.
    """
    theta = np.asarray(theta, dtype=float)
    updates = np.asarray(updates, dtype=float)
    weights = np.asarray(weights, dtype=float)
    base = validation_fpr(theta, X_val, y_val)
    limit = base + max(float(budget), 0.0)
    alphas = np.ones(len(weights))
    candidate = []
    for i, update in enumerate(updates):
        fpr = validation_fpr(theta + update, X_val, y_val)
        candidate.append(fpr)
        if fpr > limit:
            lo, hi = 0.0, 1.0
            # lo is known safe (alpha=0), hi is known unsafe (alpha=1).
            for _ in range(steps):
                mid = (lo + hi) / 2.0
                if validation_fpr(theta + mid * update, X_val, y_val) <= limit:
                    lo = mid
                else:
                    hi = mid
            alphas[i] = lo
    adjusted = weights * alphas
    if adjusted.sum() <= 0:
        adjusted = weights.copy()
    adjusted = adjusted / adjusted.sum()

    scale = 1.0
    combined = adjusted @ updates
    if validation_fpr(theta + combined, X_val, y_val) > limit:
        lo, hi = 0.0, 1.0
        for _ in range(steps):
            mid = (lo + hi) / 2.0
            if validation_fpr(theta + mid * combined, X_val, y_val) <= limit:
                lo = mid
            else:
                hi = mid
        scale = lo
    return adjusted, {
        'base_fpr': base,
        'limit': limit,
        'alphas': alphas,
        'candidate_fpr': np.asarray(candidate),
        'combined_fpr': validation_fpr(theta + scale * combined, X_val, y_val),
        'scale': float(scale),
    }
