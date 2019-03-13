import os
import sys
import pickle
import joblib

import numpy as np

from mdetsims import Sim, TEST_METADETECT_CONFIG
from metadetect.metadetect import Metadetect
from config import CONFIG


def _meas_shear(res, s2n_cut=10, trat_cut=1.2):
    op = res['1p']
    q = (
        (op['flags'] == 0) &
        (op['wmom_s2n'] > s2n_cut) &
        (op['wmom_T_ratio'] > trat_cut))
    if not np.any(q):
        return None
    g1p = op['wmom_g'][q, 0]
    g2p = op['wmom_g'][q, 1]

    om = res['1m']
    q = (
        (om['flags'] == 0) &
        (om['wmom_s2n'] > s2n_cut) &
        (om['wmom_T_ratio'] > trat_cut))
    if not np.any(q):
        return None
    g1m = om['wmom_g'][q, 0]
    g2m = om['wmom_g'][q, 1]

    o = res['noshear']
    q = (
        (o['flags'] == 0) &
        (o['wmom_s2n'] > s2n_cut) &
        (o['wmom_T_ratio'] > trat_cut))
    if not np.any(q):
        return None
    g1 = o['wmom_g'][q, 0]
    g2 = o['wmom_g'][q, 1]

    return (
        np.mean(g1p), np.mean(g1m), np.mean(g1),
        np.mean(g2p), np.mean(g2m), np.mean(g2))


def _run_sim_mdet(seed):
    config = {}
    config.update(TEST_METADETECT_CONFIG)

    rng = np.random.RandomState(seed=seed)
    mbobs = Sim(rng=rng, g1=0.02, **CONFIG).get_mbobs()
    md = Metadetect(config, mbobs, rng)
    md.go()
    pres = md.result

    rng = np.random.RandomState(seed=seed)
    mbobs = Sim(rng=rng, g1=-0.02, **CONFIG).get_mbobs()
    md = Metadetect(config, mbobs, rng)
    md.go()
    mres = md.result

    outputs = {}
    for s2n in [10, 15, 20]:
        outputs[s2n] = (
            _meas_shear(pres, s2n_cut=s2n, trat_cut=1.2),
            _meas_shear(mres, s2n_cut=s2n, trat_cut=1.2))

    return outputs


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

with open(os.path.join(odir, 'data_%05d.pkl' % seed), 'wb') as fp:
    pickle.dump(outputs, fp)
