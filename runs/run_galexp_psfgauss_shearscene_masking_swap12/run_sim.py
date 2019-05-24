import sys
import numpy as np
import schwimmbad
import multiprocessing
import logging
import time
import fitsio

from mdetsims import Sim, TEST_METACAL_MOF_CONFIG, TEST_METADETECT_CONFIG
from mdetsims.metacal import MetacalPlusMOF
from mdetsims.run_utils import (
    estimate_m_and_c, cut_nones,
    measure_shear_metadetect,
    measure_shear_metacal_plus_mof)
from metadetect.metadetect import Metadetect
from config import CONFIG

try:
    from config import SWAP12
except ImportError:
    SWAP12 = False

try:
    from config import CUT_INTERP
except ImportError:
    CUT_INTERP = False

n_sims = int(sys.argv[1])

if n_sims == 1:
    for lib in [__name__, 'ngmix', 'metadetect', 'mdetsims']:
        lgr = logging.getLogger(lib)
        hdr = logging.StreamHandler(sys.stdout)
        hdr.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
        lgr.setLevel(logging.DEBUG)
        lgr.addHandler(hdr)

LOGGER = logging.getLogger(__name__)

START = time.time()

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    HAVE_MPI = True
except Exception:
    n_ranks = 1
    rank = 0
    comm = None
    HAVE_MPI = False

if HAVE_MPI and n_ranks > 1:
    n_workers = n_ranks if n_sims > 1 else 1
else:
    n_workers = multiprocessing.cpu_count() if n_sims > 1 else 1

USE_MPI = HAVE_MPI and n_ranks > 1

try:
    from config import DO_METACAL_MOF
except Exception:
    DO_METACAL_MOF = False

if DO_METACAL_MOF:
    def _meas_shear(res):
        return measure_shear_metacal_plus_mof(
            res, s2n_cut=10, t_ratio_cut=0.5)
else:
    def _meas_shear(res):
        return measure_shear_metadetect(
            res, s2n_cut=10, t_ratio_cut=1.2, cut_interp=CUT_INTERP)


def _add_shears(cfg, plus=True):
    g1 = 0.02
    g2 = 0.0

    if not plus:
        g1 *= -1

    if SWAP12:
        g1, g2 = g2, g1

    cfg.update({'g1': g1, 'g2': g2})


def _run_sim(seed):
    if DO_METACAL_MOF:
        try:
            config = {}
            config.update(TEST_METACAL_MOF_CONFIG)

            rng = np.random.RandomState(seed=seed + 1000000)
            _add_shears(CONFIG, plus=True)
            if SWAP12:
                assert CONFIG['g1'] == 0.0
                assert CONFIG['g2'] == 0.02
            else:
                assert CONFIG['g1'] == 0.02
                assert CONFIG['g2'] == 0.0
            sim = Sim(rng=rng, **CONFIG)
            mbobs = sim.get_mbobs()
            md = MetacalPlusMOF(config, mbobs, rng)
            md.go()
            pres = _meas_shear(md.result)

            dens = len(md.result['noshear']) / sim.area_sqr_arcmin
            LOGGER.info('found %f objects per square arcminute', dens)

            rng = np.random.RandomState(seed=seed + 1000000)
            _add_shears(CONFIG, plus=False)
            if SWAP12:
                assert CONFIG['g1'] == 0.0
                assert CONFIG['g2'] == -0.02
            else:
                assert CONFIG['g1'] == -0.02
                assert CONFIG['g2'] == 0.0
            sim = Sim(rng=rng, **CONFIG)
            mbobs = sim.get_mbobs()
            md = MetacalPlusMOF(config, mbobs, rng)
            md.go()
            mres = _meas_shear(md.result)

            dens = len(md.result['noshear']) / sim.area_sqr_arcmin
            LOGGER.info('found %f objects per square arcminute', dens)

            retvals = (pres, mres)
        except Exception as e:
            print(repr(e))
            retvals = (None, None)
    else:
        try:
            config = {}
            config.update(TEST_METADETECT_CONFIG)

            rng = np.random.RandomState(seed=seed + 1000000)
            _add_shears(CONFIG, plus=True)
            if SWAP12:
                assert CONFIG['g1'] == 0.0
                assert CONFIG['g2'] == 0.02
            else:
                assert CONFIG['g1'] == 0.02
                assert CONFIG['g2'] == 0.0
            sim = Sim(rng=rng, **CONFIG)
            mbobs = sim.get_mbobs()
            md = Metadetect(config, mbobs, rng)
            md.go()
            pres = _meas_shear(md.result)

            dens = len(md.result['noshear']) / sim.area_sqr_arcmin
            LOGGER.info('found %f objects per square arcminute', dens)

            rng = np.random.RandomState(seed=seed + 1000000)
            _add_shears(CONFIG, plus=False)
            if SWAP12:
                assert CONFIG['g1'] == 0.0
                assert CONFIG['g2'] == -0.02
            else:
                assert CONFIG['g1'] == -0.02
                assert CONFIG['g2'] == 0.0
            sim = Sim(rng=rng, **CONFIG)
            mbobs = sim.get_mbobs()
            md = Metadetect(config, mbobs, rng)
            md.go()
            mres = _meas_shear(md.result)

            dens = len(md.result['noshear']) / sim.area_sqr_arcmin
            LOGGER.info('found %f objects per square arcminute', dens)

            retvals = (pres, mres)
        except Exception as e:
            print(repr(e))
            retvals = (None, None)
    if USE_MPI and seed % 1000 == 0:
        print(
            "[% 10ds] %04d: %d" % (time.time() - START, rank, seed),
            flush=True)
    return retvals


if rank == 0:
    if DO_METACAL_MOF:
        print('running metacal+MOF', flush=True)
    else:
        print('running metadetect', flush=True)
    print('config:', CONFIG, flush=True)
    print('swap 12:', SWAP12)
    print('use mpi:', USE_MPI, flush=True)
    print("n_ranks:", n_ranks, flush=True)
    print("n_workers:", n_workers, flush=True)

if not USE_MPI:
    pool = schwimmbad.JoblibPool(
        n_workers, backend='multiprocessing', verbose=100)
else:
    pool = schwimmbad.choose_pool(mpi=USE_MPI, processes=n_workers)
outputs = pool.map(_run_sim, range(n_sims))
pool.close()

pres, mres = zip(*outputs)
pres, mres = cut_nones(pres, mres)

if rank == 0:
    dt = [('g1p', 'f8'), ('g1m', 'f8'), ('g1', 'f8'),
          ('g2p', 'f8'), ('g2m', 'f8'), ('g2', 'f8')]
    dplus = np.array(pres, dtype=dt)
    dminus = np.array(mres, dtype=dt)
    with fitsio.FITS('data.fits', 'rw') as fits:
        fits.write(dplus, extname='plus')
        fits.write(dminus, extname='minus')

    m, msd, c, csd = estimate_m_and_c(pres, mres, 0.02, swap12=SWAP12)

    print("""\
# of sims: {n_sims}
noise cancel m   : {m:f} +/- {msd:f}
noise cancel c   : {c:f} +/- {csd:f}""".format(
        n_sims=len(pres),
        m=m,
        msd=msd,
        c=c,
        csd=csd), flush=True)