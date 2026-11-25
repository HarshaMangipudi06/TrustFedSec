import numpy as np
import pytest
from config import Config
from src.dataset import prepare
from src.model import train, predict_proba
from src.attacks import flip_labels, manipulate
from src.trust import aggregate, indicators, score
from src.coordinator import run

def test_train_and_predictions():
    X=np.array([[-2.],[-1.],[1.],[2.]]); y=np.array([0,0,1,1]); p=train(X,y,np.zeros(2),30,.2)
    assert predict_proba(X,p)[0] < .5 < predict_proba(X,p)[-1]

def test_aggregation():
    u=np.array([[1.,2.],[3.,4.]])
    a,w=aggregate(u,[1,3]); assert np.allclose(a,[2.5,3.5]); assert np.allclose(w,[.25,.75])
    a,w=aggregate(u,[1,3],[1,.1]); assert np.allclose(w,[1/1.3,.3/1.3]); assert np.isfinite(aggregate(u,[0,0])[0]).all()

def test_attack_and_zero_norm():
    y=np.array([0,1,0,1]); assert np.array_equal(flip_labels(y,1,np.random.default_rng(2)),1-y)
    assert np.array_equal(manipulate(np.array([1.,-2.]),2),[-2,4])
    norms,cos,dev=indicators(np.zeros((3,2))); assert np.isfinite(dev).all() and np.all(cos==1)
    t,_=score(1,100,-1,100,Config()); assert 0.1 <= t <= 1

def test_data_and_reproducibility(tmp_path):
    cfg=Config(rounds=2); a=prepare(cfg); b=prepare(cfg)
    assert all(np.array_equal(x,y) for x,y in zip(a['parts'],b['parts']))
    x=run(a,cfg,'TrustFed-Sec'); y=run(b,cfg,'TrustFed-Sec')
    assert x['history']==y['history'] and x['clients']==y['clients'] and x['test']==y['test']
    with pytest.raises(FileNotFoundError): prepare(cfg,[tmp_path/'missing.csv'])

def test_max_rows_subsamples_deterministically():
    cfg = Config(max_rows=1000)
    a = prepare(cfg); b = prepare(cfg)
    assert a['samples'] == 1000
    assert all(np.array_equal(x, y) for x, y in zip(a['parts'], b['parts']))


def test_poisoned_conditions_identical():
    cfg=Config(rounds=2,attack='label flip'); data=prepare(cfg)
    a=run(data,cfg,'Poisoned FedAvg'); b=run(data,cfg,'TrustFed-Sec')
    for x,y in zip(a['clients'][:cfg.clients],b['clients'][:cfg.clients]):
        assert x['update_norm']==y['update_norm'] and x['simulated_compromised']==y['simulated_compromised']
