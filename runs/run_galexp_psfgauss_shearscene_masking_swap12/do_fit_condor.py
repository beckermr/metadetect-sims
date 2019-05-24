import glob
import os
import pickle
import joblib

from mdetsims.run_utils import cut_nones, estimate_m_and_c

try:
    from config import SWAP12
except ImportError:
    SWAP12 = False


def _func(fname):
    try:
        with open(fname, 'rb') as fp:
            data = pickle.load(fp)
            data_10 = [d[10] for d in data]
            data_15 = [d[15] for d in data]
            data_20 = [d[20] for d in data]
        return [data_10, data_15, data_20]
    except Exception:
        return [], [], []


tmpdir = 'outputs'

if not os.path.exists('outputs/final.pkl'):
    files = glob.glob('%s/data*.pkl' % tmpdir)
    print('found %d outputs' % len(files))
    io = [joblib.delayed(_func)(fname) for fname in files]
    outputs = joblib.Parallel(
        verbose=10,
        n_jobs=-1,
        pre_dispatch='2*n_jobs',
        max_nbytes=None)(io)
else:
    print('found final output')
    with open('outputs/final.pkl', 'rb') as fp:
        outputs = pickle.load(fp)

for i, s2n in enumerate([10, 15, 20]):
    _outputs = []
    for o in outputs:
        _outputs.extend(o[i])
    pres, mres = zip(*_outputs)

    pres, mres = cut_nones(pres, mres)
    m, msd, c, csd = estimate_m_and_c(pres, mres, 0.02, swap12=SWAP12)

    print('s2n:', s2n)
    print("""\
    # of sims: {n_sims}
    m       : {m:f} +/- {msd:f}
    c       : {c:f} +/- {csd:f}""".format(
        n_sims=len(pres),
        m=m,
        msd=msd,
        c=c,
        csd=csd), flush=True)
