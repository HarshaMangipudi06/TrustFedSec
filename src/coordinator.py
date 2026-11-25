import time
import numpy as np
from dataclasses import asdict
from src.model import train
from src.attacks import flip_labels, manipulate, constrain_and_scale, harm_direction, cosine_adaptive
from src.trust import indicators, score, weights_for, fpr_budget_weights
from src.evaluation import evaluate

SCENARIOS = ('Normal FedAvg', 'Poisoned FedAvg', 'TrustFed-Sec', 'TrustFed-Sec (FPR guard)')
TRUST_SCENARIOS = ('TrustFed-Sec', 'TrustFed-Sec (FPR guard)')


def run(data, cfg, scenario, callback=None):
    if scenario not in SCENARIOS: raise ValueError('Unknown scenario')
    if not 0 <= cfg.compromised <= cfg.clients: raise ValueError('Invalid number of compromised clients')
    theta = np.zeros(data['features']+1); trusts = np.ones(cfg.clients); history=[]; clients=[]
    malicious = set(range(cfg.compromised)) if scenario != 'Normal FedAvg' else set()
    start = time.perf_counter()
    for r in range(cfg.rounds):
        updates=[None]*cfg.clients
        counts=[len(ids) for ids in data['parts']]
        for i, ids in enumerate(data['parts']):
            if i in malicious and cfg.attack == 'cosine adaptive':
                continue  # crafted below, after the honest updates are known
            y = data['y_train'][ids]
            if i in malicious and cfg.attack in ('label flip', 'constrain and scale'):
                y = flip_labels(y, cfg.poison_fraction, np.random.default_rng(np.random.SeedSequence([cfg.seed, r, i, 731])))
            update = train(data['X_train'][ids], y, theta, cfg.epochs, cfg.learning_rate)-theta
            if i in malicious and cfg.attack == 'sign flip': update = manipulate(update, cfg.strength)
            if i in malicious and cfg.attack == 'constrain and scale': update = constrain_and_scale(update, cfg.strength)
            updates[i] = update
        if malicious and cfg.attack == 'cosine adaptive':
            honest = [updates[i] for i in range(cfg.clients) if i not in malicious and updates[i] is not None]
            if honest:
                reference = np.median(np.stack(honest), axis=0)
                crafted = cosine_adaptive(reference, harm_direction(theta, data['X_val'], data['y_val']), cfg.adaptive_fraction)
                for i in malicious:
                    updates[i] = crafted.copy()
            else:
                for i in malicious:
                    ids = data['parts'][i]
                    updates[i] = train(data['X_train'][ids], data['y_train'][ids], theta, cfg.epochs, cfg.learning_rate) - theta
        norms, cos, dev = indicators(updates); baseline = evaluate(data['X_val'], data['y_val'], theta)['loss']; rows=[]
        for i, update in enumerate(updates):
            candidate = evaluate(data['X_val'], data['y_val'], theta+update); delta = candidate['loss']-baseline
            trusts[i], evidence = score(trusts[i], dev[i], cos[i], delta, cfg)
            rows.append({'round':r+1, 'client':i+1, 'samples':counts[i], 'update_norm':float(norms[i]), 'cosine':float(cos[i]), 'norm_deviation':float(dev[i]), 'validation_loss_delta':float(delta), 'candidate_attack_recall':candidate['attack_recall'], 'trust':float(trusts[i]), 'evidence':evidence, 'simulated_compromised':i in malicious, 'status':'Review' if dev[i]>cfg.norm_threshold or cos[i]<cfg.cosine_threshold or delta>cfg.validation_threshold else 'Normal'})
        # One split, one starting model: only the aggregation rule differs.
        weights = weights_for(updates, counts, trusts if scenario in TRUST_SCENARIOS else None, cfg.max_samples, cfg.trust_floor)
        guard = None
        if scenario == 'TrustFed-Sec (FPR guard)':
            weights, guard = fpr_budget_weights(theta, weights, updates, data['X_val'], data['y_val'], cfg.fpr_budget)
        step = np.asarray(updates).T @ weights
        if guard is not None:
            step = step * guard['scale']
        theta += step
        for i, row in enumerate(rows):
            row['aggregation_weight'] = float(weights[i])
            if guard is not None:
                row['fpr_alpha'] = float(guard['alphas'][i])
                row['candidate_fpr'] = float(guard['candidate_fpr'][i])
        clients.extend(rows)
        metrics=evaluate(data['X_val'], data['y_val'], theta); metrics['round']=r+1
        if guard is not None:
            metrics['val_fpr_after_guard'] = float(guard['combined_fpr'])
            metrics['guard_scale'] = float(guard['scale'])
        history.append(metrics)
        if callback: callback(r+1, rows, metrics)
    elapsed=time.perf_counter()-start
    return {'scenario':scenario, 'config':asdict(cfg), 'source':data['source'], 'samples':data['samples'], 'features':data['features'], 'history':history, 'clients':clients, 'test':evaluate(data['X_test'],data['y_test'],theta), 'seconds':elapsed, 'communication_bytes':int(cfg.rounds*cfg.clients*2*theta.size*8), 'parameters':theta.tolist()}
