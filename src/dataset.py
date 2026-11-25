import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

LEAKAGE = {'id', 'attack_cat', 'label', 'srcip', 'dstip', 'sport', 'dsport', 'source_ip', 'destination_ip'}

def prepare(config, files=None):
    if files:
        # utf-8-sig strips the BOM shipped on the official CSVs, otherwise the
        # first header becomes '\ufeffid' and the leakage filter misses it.
        frames = [pd.read_csv(f, low_memory=False, encoding='utf-8-sig') for f in files]
        df = pd.concat(frames, ignore_index=True)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if 'label' not in df.columns:
            raise ValueError('CSV must contain the binary UNSW-NB15 label column named label.')
        y = pd.to_numeric(df['label'], errors='coerce')
        if y.isna().any() or not set(y.unique()).issubset({0, 1}):
            raise ValueError('The label column must contain only binary 0/1 values.')
        y = y.to_numpy(dtype=int)
        X = df.drop(columns=[c for c in df.columns if c in LEAKAGE])
        if X.empty: raise ValueError('No usable feature columns remain.')
        source = 'UNSW-NB15 (user-supplied CSV; provenance not independently verified)'
    else:
        x, y = make_classification(n_samples=6000, n_features=18, n_informative=11, n_redundant=3, weights=[0.62, 0.38], class_sep=0.9, flip_y=0.03, random_state=config.seed)
        X = pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])])
        source = 'Synthetic demonstration data'
    if len(np.unique(y)) != 2: raise ValueError('Dataset must contain both binary classes.')
    if getattr(config, 'max_rows', None) and len(y) > config.max_rows:
        chosen = np.random.default_rng(config.seed).choice(len(y), config.max_rows, replace=False)
        X = X.iloc[chosen].reset_index(drop=True)
        y = y[chosen]
    idx = np.arange(len(y))
    train, hold = train_test_split(idx, test_size=0.30, stratify=y, random_state=config.seed)
    val, test = train_test_split(hold, test_size=0.5, stratify=y[hold], random_state=config.seed)
    nums = list(X.select_dtypes(include=np.number).columns)
    cats = [c for c in X.columns if c not in nums]
    X = X.copy()
    for c in cats: X[c] = X[c].fillna('__missing__').astype(str)
    pre = ColumnTransformer([('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), nums), ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore'))]), cats)])
    pre.fit(X.iloc[train])
    def transform(ids):
        a = pre.transform(X.iloc[ids]); return a.toarray() if hasattr(a, 'toarray') else np.asarray(a)
    xt, xv, xe = transform(train), transform(val), transform(test)
    rng = np.random.default_rng(config.seed)
    if config.clients < 2 or len(train) < config.clients * 2: raise ValueError('Not enough training samples for the selected clients.')
    if config.partition == 'IID':
        parts = list(np.array_split(rng.permutation(len(train)), config.clients))
    else:
        # Class-skewed allocation; every training sample assigned exactly once.
        buckets = [[] for _ in range(config.clients)]
        for label in (0, 1):
            ids = rng.permutation(np.flatnonzero(y[train] == label))
            proportions = rng.dirichlet(np.full(config.clients, 0.5))
            for i, chunk in enumerate(np.split(ids, np.cumsum(rng.multinomial(len(ids), proportions))[:-1])): buckets[i].extend(chunk.tolist())
        parts = [np.asarray(b, dtype=int) for b in buckets]
        if any(len(p) == 0 for p in parts): raise ValueError('Non-IID split produced an empty client; choose another seed or fewer clients.')
    return {'X_train': xt, 'y_train': y[train], 'X_val': xv, 'y_val': y[val], 'X_test': xe, 'y_test': y[test], 'parts': parts, 'source': source, 'samples': len(y), 'features': xt.shape[1]}
