import numpy as np

from scripts.bench import bootstrap_ci, contrast, holm


def test_holm_adjusts_stepwise():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert holm([1.0, 1.0]) == [1.0, 1.0]


def test_bootstrap_ci_brackets_point_estimate():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    pred = np.array([0, 0, 1, 1, 1, 1, 1, 0])
    lo, hi = bootstrap_ci(y, pred, 'accuracy', b=200, seed=1)
    assert lo <= 0.75 <= hi
    lo, hi = bootstrap_ci(y, pred, 'false_positive_rate', b=200, seed=1)
    assert lo <= 1 / 3 <= hi


def test_contrast_is_paired_by_seed():
    rows = []
    for seed in range(6):
        rows.append({'seed': seed, 'arm': 'A', 'test': {'accuracy': 0.70 + 0.01 * seed}})
        rows.append({'seed': seed, 'arm': 'B', 'test': {'accuracy': 0.60 + 0.01 * seed}})
    c = contrast(rows, 'A', 'B', 'accuracy')
    assert c['n_seeds'] == 6
    assert np.isclose(c['delta_mean'], 0.10)
    assert c['wilcoxon_p'] < 0.1


def test_bench_smoke(tmp_path):
    from scripts.bench import bench

    out = tmp_path / 'summary.json'
    s = bench(seeds=2, rounds=2, bootstrap=50, out=out, progress=False)
    assert len(s['runs']) == 8
    assert set(s['means']) == {'A_clean', 'B_poisoned', 'C_trust', 'D_guard'}
    assert all(c['holm_p'] is None or 0 <= c['holm_p'] <= 1
               for items in s['contrasts'].values() for c in items)
    assert out.exists()
