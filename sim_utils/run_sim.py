import sys
import numpy as np
import schwimmbad
import multiprocessing
import logging
import time
import fitsio

from mdetsims.metacal import (
    MetacalPlusMOF,
    MetacalTrueDetect,
    MetacalSepDetect)
from mdetsims.run_utils import (
    estimate_m_and_c, cut_nones,
    measure_shear_metadetect,
    measure_shear_metacal_plus_mof)
from metadetect.metadetect import Metadetect
from config import CONFIG

from run_preamble import get_shear_meas_config


(SWAP12, CUT_INTERP, DO_METACAL_MOF, DO_METACAL_SEP,
 DO_METACAL_TRUEDETECT,
 SHEAR_MEAS_CONFIG, SIM_CLASS) = get_shear_meas_config()

# process CLI arguments
n_sims = int(sys.argv[1])

# logging
if n_sims == 1:
    for lib in [__name__, 'ngmix', 'metadetect', 'mdetsims']:
        lgr = logging.getLogger(lib)
        hdr = logging.StreamHandler(sys.stdout)
        hdr.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
        lgr.setLevel(logging.DEBUG)
        lgr.addHandler(hdr)

LOGGER = logging.getLogger(__name__)

START = time.time()

# deal with MPI
try:
    if n_sims > 1:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        n_ranks = comm.Get_size()
        HAVE_MPI = True
    else:
        raise Exception()  # punt to the except clause
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

# code to do computation
if DO_METACAL_MOF or DO_METACAL_TRUEDETECT or DO_METACAL_SEP:
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
    config = {}
    config.update(SHEAR_MEAS_CONFIG)

    try:
        # pos shear
        rng = np.random.RandomState(seed=seed + 1000000)
        _add_shears(CONFIG, plus=True)
        if SWAP12:
            assert CONFIG['g1'] == 0.0
            assert CONFIG['g2'] == 0.02
        else:
            assert CONFIG['g1'] == 0.02
            assert CONFIG['g2'] == 0.0
        sim = SIM_CLASS(rng=rng, **CONFIG)

        if DO_METACAL_MOF:
            mbobs = sim.get_mbobs()
            md = MetacalPlusMOF(config, mbobs, rng)
            md.go()
        elif DO_METACAL_SEP:
            mbobs = sim.get_mbobs()
            md = MetacalSepDetect(config, mbobs, rng)
            md.go()
        elif DO_METACAL_TRUEDETECT:
            mbobs, tcat = sim.get_mbobs(return_truth_cat=True)
            md = MetacalTrueDetect(config, mbobs, rng, tcat)
            md.go()
        else:
            mbobs = sim.get_mbobs()
            md = Metadetect(config, mbobs, rng)
            md.go()

        pres = _meas_shear(md.result)

        dens = len(md.result['noshear']) / sim.area_sqr_arcmin
        LOGGER.info('found %f objects per square arcminute', dens)

        # neg shear
        rng = np.random.RandomState(seed=seed + 1000000)
        _add_shears(CONFIG, plus=False)
        if SWAP12:
            assert CONFIG['g1'] == 0.0
            assert CONFIG['g2'] == -0.02
        else:
            assert CONFIG['g1'] == -0.02
            assert CONFIG['g2'] == 0.0
        sim = SIM_CLASS(rng=rng, **CONFIG)

        if DO_METACAL_MOF:
            mbobs = sim.get_mbobs()
            md = MetacalPlusMOF(config, mbobs, rng)
            md.go()
        elif DO_METACAL_SEP:
            mbobs = sim.get_mbobs()
            md = MetacalSepDetect(config, mbobs, rng)
            md.go()
        elif DO_METACAL_TRUEDETECT:
            mbobs, tcat = sim.get_mbobs(return_truth_cat=True)
            md = MetacalTrueDetect(config, mbobs, rng, tcat)
            md.go()
        else:
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


if __name__ == "__main__":
    if rank == 0:
        if DO_METACAL_MOF:
            print('running metacal+MOF', flush=True)
        elif DO_METACAL_SEP:
            print('running metacal+SEP', flush=True)
        elif DO_METACAL_TRUEDETECT:
            print('running metacal+true detection', flush=True)
        else:
            print('running metadetect', flush=True)
        print('config:', CONFIG, flush=True)
        print('swap 12:', SWAP12)
        print('use mpi:', USE_MPI, flush=True)
        print("n_ranks:", n_ranks, flush=True)
        print("n_workers:", n_workers, flush=True)

    if n_workers == 1:
        outputs = [_run_sim(0)]
    else:
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
    noise cancel m [1e-3, 3-sigma]  : {m:f} +/- {msd:f}
    noise cancel c [1e-5, 3-sigma]  : {c:f} +/- {csd:f}""".format(
            n_sims=len(pres),
            m=m/1e-3,
            msd=msd*3/1e-3,
            c=c/1e-5,
            csd=csd*3/1e-5), flush=True)
