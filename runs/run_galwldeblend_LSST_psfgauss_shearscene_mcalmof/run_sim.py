import sys
import pickle
import numpy as np
import tqdm
import logging

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

DO_COMM = False

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

            return pres, mres
        except Exception as e:
            print(repr(e))
            return None, None
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

            return pres, mres
        except Exception as e:
            print(repr(e))
            return None, None


if DO_METACAL_MOF:
    print('running metacal+MOF', flush=True)
else:
    print('running metadetect', flush=True)
print('config:', CONFIG, flush=True)

outputs = []
for i in tqdm.trange(n_sims):
    if i % n_ranks == rank:
        outputs.append(_run_sim(i))
        print("%04d: %d" % (rank, i), flush=True)

pres, mres = zip(*outputs)

pres, mres = _cut(pres, mres)

print("%04d: done" % rank, flush=True)

if comm is not None and DO_COMM:
    if rank == 0:
        n_recv = 0
        while n_recv < n_ranks - 1:
            status = MPI.Status()
            data = comm.recv(
                source=MPI.ANY_SOURCE,
                tag=MPI.ANY_TAG,
                status=status)
            n_recv += 1
            pres.extend(data[0])
            mres.extend(data[1])
    else:
        comm.send((pres, mres), dest=0, tag=rank)
else:
    with open('data%04d.pkl' % rank, 'wb') as fp:
        pickle.dump((pres, mres), fp)

if HAVE_MPI:
    comm.Barrier()

if rank == 0:
    if not DO_COMM:
        print("%04d: reading rsults" % rank, flush=True)
        for i in range(1, n_ranks):
            with open('data%04d.pkl' % i, 'rb') as fp:
                data = pickle.load(fp)
                pres.extend(data[0])
                mres.extend(data[1])
        print("%04d: done reading rsults" % rank, flush=True)
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
