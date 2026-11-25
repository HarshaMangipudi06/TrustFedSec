import numpy as np
import pytest

from config import Config
from src.dataset import prepare
from src.coordinator import run
from src.trust import validation_fpr, fpr_budget_weights


def test_validation_fpr_counts_only_negative_rows():
    X = np.array([[1.0], [-1.0], [-2.0]])
    theta = np.array([2.0, 0.0])
    assert validation_fpr(theta, X, [0, 0, 1]) == 0.5
    assert validation_fpr(np.array([0.0, -1.0]), X, [0, 0, 1]) == 0.0


def test_guard_is_a_noop_when_budget_cannot_be_exceeded():
    X = np.array([[-1.0], [1.0]])
    theta = np.zeros(2)
    weights, info = fpr_budget_weights(theta, np.array([0.5, 0.5]),
                                       np.zeros((2, 2)), X, [0, 1], budget=1.0)
    assert np.allclose(weights, [0.5, 0.5])
    assert np.allclose(info['alphas'], [1.0, 1.0])
    assert info['scale'] == 1.0


def test_guard_shrinks_a_harmful_update():
    X = np.array([[-1.0], [-2.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    theta = np.array([2.0, 0.0])  # no false positives at the start
    harmful = np.array([-10.0, 0.0])  # flips both negative rows to attack
    weights, info = fpr_budget_weights(theta, np.array([0.5, 0.5]),
                                       np.array([harmful, np.zeros(2)]), X, y, budget=0.0)
    assert info['alphas'][0] < 0.2
    assert np.allclose(info['alphas'][1], 1.0)
    assert weights[0] < weights[1]
    assert info['combined_fpr'] <= info['limit'] + 1e-9


def test_guard_scenario_runs_and_records_diagnostics():
    cfg = Config(rounds=2, fpr_budget=0.02)
    data = prepare(cfg)
    result = run(data, cfg, 'TrustFed-Sec (FPR guard)')
    assert len(result['clients']) == cfg.rounds * cfg.clients
    assert all('fpr_alpha' in row for row in result['clients'])
    assert np.isfinite(result['test']['accuracy'])
    assert 'guard_scale' in result['history'][-1]
    with pytest.raises(ValueError):
        run(data, cfg, 'unknown scenario')
