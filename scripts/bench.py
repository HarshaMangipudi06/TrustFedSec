"""Multi-seed A/B/C benchmark with seed-paired statistics.

Each seed gets one split, one partition and one starting model. All three
scenarios run against that identical setup. Contrasts pair results by seed,
so a difference is not confounded by a different data split.

    python3 scripts/bench.py --seeds 30 --rounds 10

Writes results/benchmark_summary.json. Results are synthetic unless CSVs are
passed with --csv (the same loader the dashboard uses).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config
from src.dataset import prepare
from src.model import predict_proba
from src.coordinator import run

SCENARIOS = ['Normal FedAvg', 'Poisoned FedAvg', 'TrustFed-Sec', 'TrustFed-Sec (FPR guard)']
LABELS = {'Normal FedAvg': 'A_clean', 'Poisoned FedAvg': 'B_poisoned', 'TrustFed-Sec': 'C_trust',
          'TrustFed-Sec (FPR guard)': 'D_guard'}
METRICS = ['accuracy', 'false_positive_rate', 'macro_f1', 'attack_recall']
CONTRASTS = [('B_poisoned', 'A_clean'), ('C_trust', 'B_poisoned'), ('C_trust', 'A_clean'),
             ('D_guard', 'C_trust'), ('D_guard', 'A_clean')]


def bootstrap_ci(y_true, y_pred, metric, b=1000, seed=0):
    """Percentile CI over test rows. Vectorised so 30 seeds stay cheap."""
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    idx = rng.integers(0, len(yt), size=(b, len(yt)))
    yt_b, yp_b = yt[idx], yp[idx]
    if metric == 'false_positive_rate':
        neg = yt_b == 0
        fp = ((yp_b == 1) & neg).sum(axis=1)
        tn = ((yp_b == 0) & neg).sum(axis=1)
        vals = fp / np.maximum(fp + tn, 1)
    else:
        vals = (yt_b == yp_b).mean(axis=1)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted.tolist()


def contrast(rows, first, second, metric):
    """first - second, paired by seed."""
    a = {r['seed']: r for r in rows if r['arm'] == first}
    b = {r['seed']: r for r in rows if r['arm'] == second}
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 3:
        return None
    x = np.array([a[s]['test'][metric] for s in seeds])
    y = np.array([b[s]['test'][metric] for s in seeds])
    d = x - y
    n = len(d)
    sd = d.std(ddof=1)
    p_w = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    p_t = float(stats.ttest_rel(x, y).pvalue) if sd > 0 else 1.0
    se = sd / np.sqrt(n) if sd > 0 else 0.0
    margin = stats.t.ppf(0.975, n - 1) * se if se else 0.0
    return {
        'contrast': f'{first} - {second}',
        'metric': metric,
        'n_seeds': n,
        'delta_mean': float(d.mean()),
        'ci95': [float(d.mean() - margin), float(d.mean() + margin)],
        'cohens_dz': float(d.mean() / sd) if sd > 0 else None,
        'wilcoxon_p': p_w,
        'paired_t_p': p_t,
    }


def bench(seeds=30, rounds=10, attack='sign flip', bootstrap=1000,
          seed_offset=0, csv=None, out=None, progress=True, max_rows=None):
    started = time.perf_counter()
    rows = []
    for seed in range(seed_offset, seed_offset + seeds):
        cfg = Config(seed=seed, rounds=rounds, attack=attack, max_rows=max_rows)
        data = prepare(cfg, csv)
        for name in SCENARIOS:
            result = run(data, cfg, name)
            pred = (predict_proba(data['X_test'], np.asarray(result['parameters'])) >= 0.5).astype(int)
            rows.append({
                'seed': seed,
                'arm': LABELS[name],
                'scenario': name,
                'test': result['test'],
                'test_ci': {
                    'accuracy': bootstrap_ci(data['y_test'], pred, 'accuracy', bootstrap, seed),
                    'false_positive_rate': bootstrap_ci(data['y_test'], pred, 'false_positive_rate', bootstrap, seed),
                },
            })
        if progress:
            done = seed - seed_offset + 1
            vals = ' | '.join(f'{LABELS[n]} {next(r for r in rows[-len(SCENARIOS):] if r["arm"] == LABELS[n])["test"]["accuracy"]:.4f}'
                              for n in SCENARIOS)
            print(f'seed {seed + 1}/{seed_offset + seeds}: {vals}', flush=True)

    means = {
        arm: {m: float(np.mean([r['test'][m] for r in rows if r['arm'] == arm])) for m in METRICS}
        for arm in LABELS.values()
    }
    stds = {
        arm: {m: float(np.std([r['test'][m] for r in rows if r['arm'] == arm], ddof=1)) for m in METRICS}
        for arm in LABELS.values()
    }
    contrasts = {}
    for metric in ('accuracy', 'false_positive_rate'):
        items = [c for pair in CONTRASTS if (c := contrast(rows, *pair, metric)) is not None]
        adjusted = holm([c['wilcoxon_p'] for c in items])
        for c, p in zip(items, adjusted):
            c['holm_p'] = float(p)
        contrasts[metric] = items

    summary = {
        'protocol': {
            'seeds': list(range(seed_offset, seed_offset + seeds)),
            'rounds': rounds,
            'attack': attack,
            'compromised_by_default': Config().compromised,
            'clients_by_default': Config().clients,
            'partition': Config().partition,
            'source': 'user-supplied CSV' if csv else 'synthetic demonstration data',
            'max_rows': max_rows,
            'bootstrap_samples': bootstrap,
            'paired_by_seed': True,
        },
        'means': means,
        'stds': stds,
        'contrasts': contrasts,
        'runs': rows,
        'seconds': float(time.perf_counter() - started),
    }
    destination = Path(out) if out else ROOT / 'results' / 'benchmark_summary.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    if progress:
        print('\nHeld-out test means (n = %d seeds):' % seeds)
        for arm in LABELS.values():
            print(f'  {arm:12s} accuracy {means[arm]["accuracy"]:.4f} ± {stds[arm]["accuracy"]:.4f}  '
                  f'FPR {means[arm]["false_positive_rate"]:.4f} ± {stds[arm]["false_positive_rate"]:.4f}  '
                  f'F1 {means[arm]["macro_f1"]:.4f}')
        for metric, items in contrasts.items():
            print('\n' + metric + ':')
            for c in items:
                print(f'  {c["contrast"]:32s} delta {c["delta_mean"]:+.4f}  '
                      f'd_z {c["cohens_dz"] if c["cohens_dz"] is not None else float("nan"):+.3f}  '
                      f'wilcoxon {c["wilcoxon_p"]:.4f}  holm {c["holm_p"]:.4f}')
        print(f'\nWrote {destination} in {summary["seconds"]:.1f}s')
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--attack', default='sign flip',
                    choices=['sign flip', 'label flip', 'constrain and scale', 'cosine adaptive'])
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--seed-offset', type=int, default=0)
    ap.add_argument('--csv', nargs='*', default=None)
    ap.add_argument('--max-rows', type=int, default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    bench(a.seeds, a.rounds, a.attack, a.bootstrap, a.seed_offset, a.csv, a.out, max_rows=a.max_rows)


if __name__ == '__main__':
    main()
