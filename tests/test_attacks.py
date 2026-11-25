import numpy as np
import pytest

from config import Config
from src.attacks import constrain_and_scale, harm_direction, cosine_adaptive, flip_labels
from src.coordinator import run
from src.dataset import prepare
from src.evaluation import evaluate


def test_constrain_and_scale_multiplies_update():
    update = np.array([1.0, -2.0])
    assert np.allclose(constrain_and_scale(update, 5), [5.0, -10.0])


def test_harm_direction_points_toward_higher_loss():
    X = np.array([[-1.0], [1.0]])
    y = np.array([0, 1])
    theta = np.zeros(2)
    before = evaluate(X, y, theta)['loss']
    after = evaluate(X, y, theta + 0.5 * harm_direction(theta, X, y))['loss']
    assert after > before


def test_cosine_adaptive_stays_close_to_median():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20)
    harm = rng.normal(size=20)
    delta = cosine_adaptive(ref, harm, fraction=0.05)
    assert delta.shape == ref.shape
    cos = np.dot(delta, ref) / (np.linalg.norm(delta) * np.linalg.norm(ref))
    assert cos > 0.95
    assert np.linalg.norm(delta) == pytest.approx(np.linalg.norm(ref), rel=1e-6)


def test_new_attacks_run_and_are_deterministic():
    for attack in ['constrain and scale', 'cosine adaptive']:
        cfg = Config(rounds=2, attack=attack)
        data = prepare(cfg)
        a = run(data, cfg, 'TrustFed-Sec')
        b = run(data, cfg, 'TrustFed-Sec')
        assert a['history'] == b['history']
        assert np.isfinite(a['test']['accuracy'])
        assert len(a['clients']) == cfg.rounds * cfg.clients
