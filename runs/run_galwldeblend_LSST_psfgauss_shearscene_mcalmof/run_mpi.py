import sys
import pickle
import logging

from config import CONFIG
from run_sim import run_sims, _cut, _fit_m


try:
    from config import DO_METACAL_MOF
except Exception:
    DO_METACAL_MOF = False

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

if DO_METACAL_MOF:
    print('running metacal+MOF', flush=True)
else:
    print('running metadetect', flush=True)
print('config:', CONFIG, flush=True)

n_sims = int(sys.argv[1])
outputs = run_sims(rank, n_sims)


pres, mres = zip(*outputs)

pres, mres = _cut(pres, mres)

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
        comm.send((pres, mres), dest=0, tag=0)
else:
    with open('data%d.pkl' % rank, 'wb') as fp:
        pickle.dump((pres, mres), fp)

if HAVE_MPI:
    comm.Barrier()

if rank == 0:
    if not DO_COMM:
        for i in range(1, n_ranks):
            with open('data%d.pkl' % i, 'rb') as fp:
                data = pickle.load(fp)
                pres.extend(data[0])
                mres.extend(data[1])

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
