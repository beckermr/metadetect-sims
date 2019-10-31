import glob
import pickle
import joblib
import numpy as np

from mdetsims.run_utils import cut_nones, estimate_m_and_c

try:
    from config import SWAP12
except ImportError:
    SWAP12 = False


def _cut_nones_and_avg_data(data):
    nd = [d for d in data if d[0] is not None and d[1] is not None]
    if len(nd) == 0:
        return None, 0
    else:
        return np.mean(np.array(nd), axis=0).tolist(), len(nd)


def _func(fname):
    try:
        with open(fname, 'rb') as fp:
            data = pickle.load(fp)
            data_10 = [d[10] for d in data]
            data_15 = [d[15] for d in data]
            data_20 = [d[20] for d in data]
    except Exception:
        return None, None, None

    return (
        _cut_nones_and_avg_data(data_10),
        _cut_nones_and_avg_data(data_15),
        _cut_nones_and_avg_data(data_20))


tmpdir = 'outputs'

files = glob.glob('%s/data*.pkl' % tmpdir)
print('found %d outputs' % len(files))
io = [joblib.delayed(_func)(fname) for fname in files]
outputs = joblib.Parallel(
    verbose=10,
    n_jobs=-1,
    pre_dispatch='2*n_jobs',
    max_nbytes=None)(io)

for i, s2n in enumerate([10, 15, 20]):
    _outputs = [o[i][0] for o in outputs if o[i][0] is not None]
    _wgts = [o[i][1] for o in outputs if o[i][0] is not None]
    pres, mres = zip(*_outputs)

    pres, mres = cut_nones(pres, mres)
    m, msd, c, csd = estimate_m_and_c(
        pres, mres, 0.02, swap12=SWAP12, weights=_wgts)

    if np.abs(m) < 1e-2:
        mfac = 1e-3
        mstr = '[1e-3]'
    else:
        mfac = 1
        mstr = '      '

    if np.abs(c) < 1e-4:
        cfac = 1e-4
        cstr = '[1e-4]'
    else:
        cfac = 1
        cstr = '      '

    print('s2n:', s2n)
    print("""\
    # of sims: {n_sims}
    m {mstr:s}: {m:f} +/- {msd:f}
    c {cstr:s}: {c:f} +/- {csd:f}""".format(
        n_sims=len(pres) * np.sum(_wgts),
        mstr=mstr,
        m=m/mfac,
        msd=msd/mfac,
        cstr=cstr,
        c=c/cfac,
        csd=csd/cfac), flush=True)
