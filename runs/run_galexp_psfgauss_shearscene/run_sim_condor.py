import os
import sys
import pickle
import joblib

import numpy as np

from test_sim_utils import Sim, TEST_METADETECT_CONFIG
from metadetect.metadetect import Metadetect
from config import CONFIG


def _meas_shear(res):
    op = res['1p']
    q = (op['flags'] == 0) & (op['wmom_s2n'] > 10) & (op['wmom_T_ratio'] > 1.2)
    if not np.any(q):
        return None
    g1p = op['wmom_g'][q, 0]

    om = res['1m']
    q = (om['flags'] == 0) & (om['wmom_s2n'] > 10) & (om['wmom_T_ratio'] > 1.2)
    if not np.any(q):
        return None
    g1m = om['wmom_g'][q, 0]

    o = res['noshear']
    q = (o['flags'] == 0) & (o['wmom_s2n'] > 10) & (o['wmom_T_ratio'] > 1.2)
    if not np.any(q):
        return None
    g1 = o['wmom_g'][q, 0]

    return np.mean(g1p), np.mean(g1m), np.mean(g1)


def _cut(prr, mrr):
    prr_keep = []
    mrr_keep = []
    for pr, mr in zip(prr, mrr):
        if pr is None or mr is None:
            continue
        prr_keep.append(pr)
        mrr_keep.append(mr)
    return prr_keep, mrr_keep


def _get_stuff(rr):
    _a = np.vstack(rr)
    g1p = _a[:, 0]
    g1m = _a[:, 1]
    g1 = _a[:, 2]

    return g1, (g1p - g1m) / 2 / 0.01 * 0.02


def _fit_m(prr, mrr):
    g1p, R11p = _get_stuff(prr)
    g1m, R11m = _get_stuff(mrr)

    x = (R11p + R11m)/2
    y = (g1p - g1m)/2

    rng = np.random.RandomState(seed=100)
    mvals = []
    for _ in range(10000):
        ind = rng.choice(len(y), replace=True, size=len(y))
        mvals.append(np.mean(y[ind]) / np.mean(x[ind]) - 1)

    return np.mean(y) / np.mean(x) - 1, np.std(mvals)


def _fit_m_single(prr):
    g1p, R11p = _get_stuff(prr)

    x = R11p
    y = g1p

    rng = np.random.RandomState(seed=100)
    mvals = []
    for _ in range(10000):
        ind = rng.choice(len(y), replace=True, size=len(y))
        mvals.append(np.mean(y[ind]) / np.mean(x[ind]) - 1)

    return np.mean(y) / np.mean(x) - 1, np.std(mvals)


def _run_sim_mdet(seed):
    config = {}
    config.update(TEST_METADETECT_CONFIG)

    rng = np.random.RandomState(seed=seed)
    mbobs = Sim(rng=rng, g1=0.02, **CONFIG).get_mbobs()
    md = Metadetect(config, mbobs, rng)
    md.go()
    pres = _meas_shear(md.result)

    rng = np.random.RandomState(seed=seed)
    mbobs = Sim(rng=rng, g1=-0.02, **CONFIG).get_mbobs()
    md = Metadetect(config, mbobs, rng)
    md.go()
    mres = _meas_shear(md.result)

    return pres, mres


print('running metadetect', flush=True)
print('config:', CONFIG, flush=True)

n_sims = int(sys.argv[1])
seed = int(sys.argv[2])
odir = sys.argv[3]

seeds = np.random.RandomState(seed).randint(
    low=0,
    high=2**30,
    size=n_sims)

sims = [joblib.delayed(_run_sim_mdet)(s) for s in seeds]
outputs = joblib.Parallel(
    verbose=100,
    n_jobs=1,
    pre_dispatch='1',
    max_nbytes=None)(sims)

pres, mres = zip(*outputs)
pres, mres = _cut(pres, mres)

with open(os.path.join(odir, 'data_%05d.pkl' % seed), 'wb') as fp:
    pickle.dump((pres, mres), fp)
