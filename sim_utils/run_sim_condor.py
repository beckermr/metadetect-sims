import os
import sys
import pickle
import joblib

import numpy as np

from mdetsims import Sim, TEST_METADETECT_CONFIG
from metadetect.metadetect import Metadetect
from config import CONFIG


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

with open(os.path.join(odir, 'data_%05d.pkl' % seed), 'wb') as fp:
    pickle.dump(outputs, fp)
