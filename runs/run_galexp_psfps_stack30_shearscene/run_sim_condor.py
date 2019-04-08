import os
import sys
import pickle
import joblib

import numpy as np

from mdetsims import Sim, TEST_METACAL_MOF_CONFIG, TEST_METADETECT_CONFIG
from mdetsims.metacal import MetacalPlusMOF, METACAL_TYPES
from metadetect.metadetect import Metadetect
from config import CONFIG

try:
    from config import DO_METACAL_MOF
except Exception:
    DO_METACAL_MOF = False

if DO_METACAL_MOF:
    def _mask(mof, cat, *, s2n_cut, trat_cut):
        return (
            (mof['flags'] == 0) &
            (cat['mcal_s2n'] > s2n_cut) &
            (cat['mcal_T_ratio'] > trat_cut))

    def _meas_shear(res, s2n_cut=10, trat_cut=0.5):
        msks = {}
        for sh in METACAL_TYPES:
            msks[sh] = _mask(
                res['mof'], res[sh], s2n_cut=s2n_cut, trat_cut=trat_cut)
            if not np.any(msks[sh]):
                return None

        g1p = res['1p']['mcal_g'][msks['1p'], 0]
        g1m = res['1m']['mcal_g'][msks['1m'], 0]

        g2p = res['2p']['mcal_g'][msks['2p'], 1]
        g2m = res['2m']['mcal_g'][msks['2m'], 1]

        g1 = res['noshear']['mcal_g'][msks['noshear'], 0]
        g2 = res['noshear']['mcal_g'][msks['noshear'], 1]

        return (
            np.mean(g1p), np.mean(g1m), np.mean(g1),
            np.mean(g2p), np.mean(g2m), np.mean(g2))
else:
    def _meas_shear(res, s2n_cut=10, trat_cut=1.2):
        op = res['1p']
        q = (
            (op['flags'] == 0) &
            (op['wmom_s2n'] > s2n_cut) &
            (op['wmom_T_ratio'] > trat_cut))
        if not np.any(q):
            return None
        g1p = op['wmom_g'][q, 0]

        om = res['1m']
        q = (
            (om['flags'] == 0) &
            (om['wmom_s2n'] > s2n_cut) &
            (om['wmom_T_ratio'] > trat_cut))
        if not np.any(q):
            return None
        g1m = om['wmom_g'][q, 0]

        o = res['noshear']
        q = (
            (o['flags'] == 0) &
            (o['wmom_s2n'] > s2n_cut) &
            (o['wmom_T_ratio'] > trat_cut))
        if not np.any(q):
            return None
        g1 = o['wmom_g'][q, 0]
        g2 = o['wmom_g'][q, 1]

        op = res['2p']
        q = (
            (op['flags'] == 0) &
            (op['wmom_s2n'] > s2n_cut) &
            (op['wmom_T_ratio'] > trat_cut))
        g2p = op['wmom_g'][q, 1]

        om = res['2m']
        q = (
            (om['flags'] == 0) &
            (om['wmom_s2n'] > s2n_cut) &
            (om['wmom_T_ratio'] > trat_cut))
        if not np.any(q):
            return None
        g2m = om['wmom_g'][q, 1]

        return (
            np.mean(g1p), np.mean(g1m), np.mean(g1),
            np.mean(g2p), np.mean(g2m), np.mean(g2))


def _run_sim(seed):
    if DO_METACAL_MOF:
        config = {}
        config.update(TEST_METACAL_MOF_CONFIG)

        rng = np.random.RandomState(seed=seed)
        mbobs = Sim(rng=rng, g1=0.02, **CONFIG).get_mbobs()
        md = MetacalPlusMOF(config, mbobs, rng)
        md.go()
        pres = md.result

        rng = np.random.RandomState(seed=seed)
        mbobs = Sim(rng=rng, g1=-0.02, **CONFIG).get_mbobs()
        md = MetacalPlusMOF(config, mbobs, rng)
        md.go()
        mres = md.result

    else:
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
            _meas_shear(pres, s2n_cut=s2n),
            _meas_shear(mres, s2n_cut=s2n))

    return outputs


if DO_METACAL_MOF:
    print('running metacal+MOF', flush=True)
else:
    print('running metadetect', flush=True)
print('config:', CONFIG, flush=True)

n_sims = int(sys.argv[1])
seed = int(sys.argv[2])
odir = sys.argv[3]

seeds = np.random.RandomState(seed).randint(
    low=0,
    high=2**30,
    size=n_sims)

sims = [joblib.delayed(_run_sim)(s) for s in seeds]
outputs = joblib.Parallel(
    verbose=100,
    n_jobs=1,
    pre_dispatch='1',
    max_nbytes=None)(sims)

with open(os.path.join(odir, 'data_%05d.pkl' % seed), 'wb') as fp:
    pickle.dump(outputs, fp)
