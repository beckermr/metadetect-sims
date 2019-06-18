import os
import sys
import pickle
import joblib

import numpy as np

from mdetsims import (
    Sim, TEST_METACAL_MOF_CONFIG, TEST_METADETECT_CONFIG,
    TEST_METACAL_TRUEDETECT_CONFIG)
from mdetsims.metacal import MetacalPlusMOF, MetacalTrueDetect
from mdetsims.run_utils import (
    measure_shear_metadetect, measure_shear_metacal_plus_mof)
from metadetect.metadetect import Metadetect
from config import CONFIG

try:
    from config import CUT_INTERP
except ImportError:
    CUT_INTERP = False

try:
    from config import DO_METACAL_MOF
except Exception:
    DO_METACAL_MOF = False

try:
    from config import DO_METACAL_TRUEDETECT
except Exception:
    DO_METACAL_TRUEDETECT = False

try:
    from config import SWAP12
except ImportError:
    SWAP12 = False

try:
    from config import EXTRA_MDET_CONFIG
    TEST_METADETECT_CONFIG.update(EXTRA_MDET_CONFIG)
except ImportError:
    pass

if DO_METACAL_MOF or DO_METACAL_TRUEDETECT:
    def _meas_shear(res, *, s2n_cut):
        return measure_shear_metacal_plus_mof(
            res, s2n_cut=s2n_cut, t_ratio_cut=0.5)
else:
    def _meas_shear(res, *, s2n_cut):
        return measure_shear_metadetect(
            res, s2n_cut=s2n_cut, t_ratio_cut=1.2,
            cut_interp=CUT_INTERP)


def _add_shears(cfg, plus=True):
    g1 = 0.02
    g2 = 0.0

    if not plus:
        g1 *= -1

    if SWAP12:
        g1, g2 = g2, g1

    cfg.update({'g1': g1, 'g2': g2})


def _run_sim(seed):
    try:
        if DO_METACAL_MOF:
            config = {}
            config.update(TEST_METACAL_MOF_CONFIG)

            rng = np.random.RandomState(seed=seed)
            _add_shears(CONFIG, plus=True)
            if SWAP12:
                assert CONFIG['g1'] == 0.0
                assert CONFIG['g2'] == 0.02
            else:
                assert CONFIG['g1'] == 0.02
                assert CONFIG['g2'] == 0.0
            mbobs = Sim(rng=rng, **CONFIG).get_mbobs()
            md = MetacalPlusMOF(config, mbobs, rng)
            md.go()
            pres = md.result

            rng = np.random.RandomState(seed=seed)
            _add_shears(CONFIG, plus=False)
            if SWAP12:
                assert CONFIG['g1'] == 0.0
                assert CONFIG['g2'] == -0.02
            else:
                assert CONFIG['g1'] == -0.02
                assert CONFIG['g2'] == 0.0
            mbobs = Sim(rng=rng, **CONFIG).get_mbobs()
            md = MetacalPlusMOF(config, mbobs, rng)
            md.go()
            mres = md.result
        elif DO_METACAL_TRUEDETECT:
            config = {}
            config.update(TEST_METACAL_TRUEDETECT_CONFIG)

            rng = np.random.RandomState(seed=seed)
            _add_shears(CONFIG, plus=True)
            if SWAP12:
                assert CONFIG['g1'] == 0.0
                assert CONFIG['g2'] == 0.02
            else:
                assert CONFIG['g1'] == 0.02
                assert CONFIG['g2'] == 0.0
            sim = Sim(rng=rng, **CONFIG)
            mbobs, tcat = sim.get_mbobs(return_truth_cat=True)
            md = MetacalTrueDetect(config, mbobs, rng, tcat)
            md.go()
            pres = md.result

            rng = np.random.RandomState(seed=seed)
            _add_shears(CONFIG, plus=False)
            if SWAP12:
                assert CONFIG['g1'] == 0.0
                assert CONFIG['g2'] == -0.02
            else:
                assert CONFIG['g1'] == -0.02
                assert CONFIG['g2'] == 0.0
            sim = Sim(rng=rng, **CONFIG)
            mbobs, tcat = sim.get_mbobs(return_truth_cat=True)
            md = MetacalTrueDetect(config, mbobs, rng, tcat)
            md.go()
            mres = md.result
        else:
            config = {}
            config.update(TEST_METADETECT_CONFIG)

            rng = np.random.RandomState(seed=seed)
            _add_shears(CONFIG, plus=True)
            if SWAP12:
                assert CONFIG['g1'] == 0.0
                assert CONFIG['g2'] == 0.02
            else:
                assert CONFIG['g1'] == 0.02
                assert CONFIG['g2'] == 0.0
            mbobs = Sim(rng=rng, **CONFIG).get_mbobs()
            md = Metadetect(config, mbobs, rng)
            md.go()
            pres = md.result

            rng = np.random.RandomState(seed=seed)
            _add_shears(CONFIG, plus=False)
            if SWAP12:
                assert CONFIG['g1'] == 0.0
                assert CONFIG['g2'] == -0.02
            else:
                assert CONFIG['g1'] == -0.02
                assert CONFIG['g2'] == 0.0
            mbobs = Sim(rng=rng, **CONFIG).get_mbobs()
            md = Metadetect(config, mbobs, rng)
            md.go()
            mres = md.result

        outputs = {}
        for s2n in [10, 15, 20]:
            outputs[s2n] = (
                _meas_shear(pres, s2n_cut=s2n),
                _meas_shear(mres, s2n_cut=s2n))

        return outputs

    except Exception as e:
        print(repr(e))
        return {10: (None, None), 15: (None, None), 20: (None, None)}


if DO_METACAL_MOF:
    print('running metacal+MOF', flush=True)
elif DO_METACAL_TRUEDETECT:
    print('running metacal+true detection', flush=True)
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
