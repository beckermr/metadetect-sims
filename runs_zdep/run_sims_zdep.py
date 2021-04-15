import sys
import numpy as np
import schwimmbad
import multiprocessing
import logging
import time
import fitsio
from mdetsims.run_utils import (
    estimate_m_and_c, cut_nones,
    measure_shear_metadetect)
from metadetect.metadetect import Metadetect
from config import CONFIG
from config import match, invert_ex
from run_preamble_new import get_shear_meas_config

(SWAP12, CUT_INTERP,
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
def _meas_shear(res):
    return measure_shear_metadetect(
        res, s2n_cut=10, t_ratio_cut=1.2, cut_interp=CUT_INTERP)


def _add_shears(cfg, plus=True):
    g1 = 0.02
    g2 = 0.0
    g1ex = -0.02
    g2ex = 0.0

    if not plus:
        g1 *= -1

    if not plus and invert_ex:
        g1ex *= -1

    if SWAP12:
        g1, g2 = g2, g1
        g1ex, g2ex = g2ex, g1ex

    cfg.update({'g1': g1, 'g2': g2, 'g1ex': g1ex, 'g2ex': g2ex})


def gal_match(pos_det, truth_cat):

    gal_det = []
    gal_sim = []
    pos_sim_x = truth_cat['x']
    pos_sim_y = truth_cat['y']
    z_population = truth_cat['z_population']

    for i in range(len(pos_det)):
        gal_det.append([pos_det[i][-9], pos_det[i][-10]])
    for i in range(len(pos_sim_x)):
        gal_sim.append((pos_sim_x[i], pos_sim_y[i]))

    from scipy.spatial import cKDTree
    tree = cKDTree(gal_sim)
    index = []
    shear_det = []

    for i in range(len(gal_det)):
        galaxy = gal_det[i]
        _, idx = tree.query((galaxy[0], galaxy[1]))
        # use query to get the cloest match
        index.append(idx)
        shear_det.append(z_population[idx])

    index, shear_det = np.array(index), np.array(shear_det)
    return index, shear_det


def make_catalog(res, truth_cat, pop_index):

    dic = {}
    _, shear_det = gal_match(res['noshear'], truth_cat)
    dic['noshear'] = res['noshear'][np.where((shear_det == pop_index),
                                             True, False)]
    _, shear_det = gal_match(res['1p'], truth_cat)
    dic['1p'] = res['1p'][np.where((shear_det == pop_index), True, False)]
    _, shear_det = gal_match(res['1m'], truth_cat)
    dic['1m'] = res['1m'][np.where((shear_det == pop_index), True, False)]
    _, shear_det = gal_match(res['2p'], truth_cat)
    dic['2p'] = res['2p'][np.where((shear_det == pop_index), True, False)]
    _, shear_det = gal_match(res['2m'], truth_cat)
    dic['2m'] = res['2m'][np.where((shear_det == pop_index), True, False)]

    return dic


def _run_sim(seed):
    config = {}
    config.update(SHEAR_MEAS_CONFIG)

    try:
        # pos shear
        rng = np.random.RandomState(seed=seed+1000000)
        sel_rng = np.random.RandomState(seed=seed+1000000)
        _add_shears(CONFIG, plus=True)
        if SWAP12:
            assert CONFIG['g1'] == 0.0
            assert CONFIG['g2'] == 0.02
            assert CONFIG['g1ex'] == 0.0
            assert CONFIG['g2ex'] == -0.02
        else:
            assert CONFIG['g1'] == 0.02
            assert CONFIG['g2'] == 0.0
            assert CONFIG['g1ex'] == -0.02
            assert CONFIG['g2ex'] == 0.0
        sim = SIM_CLASS(rng=rng, sel_rng=sel_rng, frac_ex=0.5, **CONFIG)

        mbobs, truth_cat = sim.get_mbobs(return_truth_cat=True)
        md = Metadetect(config, mbobs, rng)
        md.go()

        if match:
            gal = make_catalog(md.result, truth_cat, 1)
            pres = _meas_shear(gal)
            gal_ex = make_catalog(md.result, truth_cat, 0)
            pres_ex = _meas_shear(gal_ex)
        else:
            pres = _meas_shear(md.result)

        dens = len(md.result['noshear']) / sim.area_sqr_arcmin
        LOGGER.info('found %f objects per square arcminute', dens)

        # neg shear
        rng = np.random.RandomState(seed=seed+1000000)
        sel_rng = np.random.RandomState(seed=seed+1000000)
        _add_shears(CONFIG, plus=False)
        if SWAP12:
            assert CONFIG['g1'] == 0.0
            assert CONFIG['g2'] == -0.02
            assert CONFIG['g1ex'] == 0.0
            assert CONFIG['g2ex'] == 0.02
        else:
            assert CONFIG['g1'] == -0.02
            assert CONFIG['g2'] == 0.0
            assert CONFIG['g1ex'] == 0.02
            assert CONFIG['g2ex'] == 0.0
        sim = SIM_CLASS(rng=rng, sel_rng=sel_rng, frac_ex=0.5, **CONFIG)

        mbobs, truth_cat = sim.get_mbobs(return_truth_cat=True)
        md = Metadetect(config, mbobs, rng)
        md.go()

        if match:
            gal = make_catalog(md.result, truth_cat, 1)
            mres = _meas_shear(gal)
            gal_ex = make_catalog(md.result, truth_cat, 0)
            mres_ex = _meas_shear(gal_ex)
            retvals = (pres, mres, pres_ex, mres_ex)
        else:
            mres = _meas_shear(md.result)
            retvals = (pres, mres)

        dens = len(md.result['noshear']) / sim.area_sqr_arcmin
        LOGGER.info('found %f objects per square arcminute', dens)

    except Exception as e:
        print(repr(e))
        retvals = (None, None)

    if USE_MPI and seed % 10000 == 0:
        print("[% 10ds] %04d: %d" % (time.time() - START, rank, seed),
              flush=True)
    return retvals


if rank == 0:
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

if match:
    pres, mres, pres_ex, mres_ex = zip(*outputs)
    pres, mres = cut_nones(pres, mres)
    pres_ex, mres_ex = cut_nones(pres_ex, mres_ex)

    if rank == 0:
        dt = [('g1p', 'f8'), ('g1m', 'f8'), ('g1', 'f8'),
              ('g2p', 'f8'), ('g2m', 'f8'), ('g2', 'f8')]
        dplus = np.array(pres, dtype=dt)
        dminus = np.array(mres, dtype=dt)
        dplus_ex = np.array(pres_ex, dtype=dt)
        dminus_ex = np.array(mres_ex, dtype=dt)

        with fitsio.FITS('data.fits', 'rw') as fits:
            fits.write(dplus, extname='plus')
            fits.write(dminus, extname='minus')
            fits.write(dplus_ex, extname='plus_ex')
            fits.write(dminus_ex, extname='minus_ex')

        m, msd, c, csd = estimate_m_and_c(pres, mres, 0.02, swap12=SWAP12)
        m_ex, msd_ex, c_ex, csd_ex = estimate_m_and_c(pres_ex, mres_ex,
                                                      0.02, swap12=SWAP12)

        print("""\
    # of sims: {n_sims}
    noise cancel m   : {m:f} +/- {msd:f}
    noise cancel c   : {c:f} +/- {csd:f}""".format(
            n_sims=len(pres),
            m=m,
            msd=msd,
            c=c,
            csd=csd), flush=True)

        print("""\
    # of sims: {n_sims}
    noise cancel m   : {m:f} +/- {msd:f}
    noise cancel c   : {c:f} +/- {csd:f}""".format(
            n_sims=len(pres_ex),
            m=m_ex,
            msd=msd_ex,
            c=c_ex,
            csd=csd_ex), flush=True)

else:
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
