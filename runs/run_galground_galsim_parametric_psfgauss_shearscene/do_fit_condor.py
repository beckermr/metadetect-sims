import glob
import os
import pickle
import tqdm
import numpy as np
import joblib


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

    msk = (
        np.isfinite(g1p) &
        np.isfinite(R11p) &
        np.isfinite(g1m) &
        np.isfinite(R11m) &
        np.isfinite(g2p) &
        np.isfinite(R22p) &
        np.isfinite(g2m) &
        np.isfinite(R22m))
    g1p = g1p[msk]
    R11p = R11p[msk]
    g1m = g1m[msk]
    R11m = R11m[msk]
    g2p = g2p[msk]
    R22p = R22p[msk]
    g2m = g2m[msk]
    R22m = R22m[msk]

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

    pres, mres = _cut(pres, mres)
    m, msd, c, csd = _fit_m(pres, mres)

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
