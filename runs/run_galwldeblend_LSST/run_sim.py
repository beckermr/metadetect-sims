import sys
import numpy as np
import pickle
import tqdm
import schwimmbad
import multiprocessing
import logging
import time

from mdetsims import Sim, TEST_METACAL_MOF_CONFIG, TEST_METADETECT_CONFIG
from mdetsims.metacal import MetacalPlusMOF, METACAL_TYPES
from metadetect.metadetect import Metadetect
from config import CONFIG

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
    def _mask(mof, cat, s2n_cut=10, size_cut=0.5):
        return (
            (mof['flags'] == 0) &
            (cat['mcal_s2n'] > s2n_cut) &
            (cat['mcal_T_ratio'] > size_cut))

    def _meas_shear(res):
        msks = {}
        for sh in METACAL_TYPES:
            msks[sh] = _mask(res['mof'], res[sh])
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
    def _meas_shear(res):
        op = res['1p']
        q = ((op['flags'] == 0) &
             (op['wmom_s2n'] > 10) &
             (op['wmom_T_ratio'] > 1.2))
        if not np.any(q):
            return None
        g1p = op['wmom_g'][q, 0]

        om = res['1m']
        q = ((om['flags'] == 0) &
             (om['wmom_s2n'] > 10) &
             (om['wmom_T_ratio'] > 1.2))
        if not np.any(q):
            return None
        g1m = om['wmom_g'][q, 0]

        o = res['noshear']
        q = ((o['flags'] == 0) &
             (o['wmom_s2n'] > 10) &
             (o['wmom_T_ratio'] > 1.2))
        if not np.any(q):
            return None
        g1 = o['wmom_g'][q, 0]
        g2 = o['wmom_g'][q, 1]

        op = res['2p']
        q = ((op['flags'] == 0) &
             (op['wmom_s2n'] > 10) &
             (op['wmom_T_ratio'] > 1.2))
        if not np.any(q):
            return None
        g2p = op['wmom_g'][q, 1]

        op = res['2m']
        q = ((op['flags'] == 0) &
             (op['wmom_s2n'] > 10) &
             (op['wmom_T_ratio'] > 1.2))
        if not np.any(q):
            return None
        g2m = op['wmom_g'][q, 1]

        return (
            np.mean(g1p), np.mean(g1m), np.mean(g1),
            np.mean(g2p), np.mean(g2m), np.mean(g2))


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
    g2p = _a[:, 3]
    g2m = _a[:, 4]
    g2 = _a[:, 5]

    return (
        g1, (g1p - g1m) / 2 / 0.01 * 0.02,
        g2, (g2p - g2m) / 2 / 0.01)


def _fit_m(prr, mrr):
    g1p, R11p, g2p, R22p = _get_stuff(prr)
    g1m, R11m, g2m, R22m = _get_stuff(mrr)

    x1 = (R11p + R11m)/2
    y1 = (g1p - g1m) / 2

    x2 = (R22p + R22m) / 2
    y2 = (g2p + g2m) / 2

    rng = np.random.RandomState(seed=100)
    mvals = []
    cvals = []
    for _ in tqdm.trange(500, leave=False):
        ind = rng.choice(len(y1), replace=True, size=len(y1))
        mvals.append(np.mean(y1[ind]) / np.mean(x1[ind]) - 1)
        cvals.append(np.mean(y2[ind]) / np.mean(x2[ind]))

    return (
        np.mean(y1) / np.mean(x1) - 1, np.std(mvals),
        np.mean(y2) / np.mean(x2), np.std(cvals))


def _run_sim(seed):
    if DO_METACAL_MOF:
        try:
            config = {}
            config.update(TEST_METACAL_MOF_CONFIG)

            rng = np.random.RandomState(seed=seed + 1000000)
            sim = Sim(rng=rng, g1=0.02, **CONFIG)
            mbobs = sim.get_mbobs()
            md = MetacalPlusMOF(config, mbobs, rng)
            md.go()
            pres = _meas_shear(md.result)

            dens = len(md.result['noshear']) / sim.area_sqr_arcmin
            LOGGER.info('found %f objects per square arcminute', dens)

            rng = np.random.RandomState(seed=seed + 1000000)
            sim = Sim(rng=rng, g1=-0.02, **CONFIG)
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
            sim = Sim(rng=rng, g1=0.02, **CONFIG)
            mbobs = sim.get_mbobs()
            md = Metadetect(config, mbobs, rng)
            md.go()
            pres = _meas_shear(md.result)

            dens = len(md.result['noshear']) / sim.area_sqr_arcmin
            LOGGER.info('found %f objects per square arcminute', dens)

            rng = np.random.RandomState(seed=seed + 1000000)
            sim = Sim(rng=rng, g1=-0.02, **CONFIG)
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
    if USE_MPI:
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
pres, mres = _cut(pres, mres)

if rank == 0:
    with open('data.pkl', 'wb') as fp:
        pickle.dump(outputs, fp)
    m, msd, c, csd = _fit_m(pres, mres)

    print("""\
# of sims: {n_sims}
noise cancel m   : {m:f} +/- {msd:f}
noise cancel c   : {c:f} +/- {csd:f}""".format(
        n_sims=len(pres),
        m=m,
        msd=msd,
        c=c,
        csd=csd), flush=True)