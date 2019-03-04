import glob
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

    return g1, (g1p - g1m) / 2 / 0.01 * 0.02


def _fit_m(prr, mrr):
    g1p, R11p = _get_stuff(prr)
    g1m, R11m = _get_stuff(mrr)

    x = R11p + R11m
    y = g1p - g1m

    rng = np.random.RandomState(seed=100)
    mvals = []
    for _ in tqdm.trange(1000, leave=False):
        ind = rng.choice(len(y), replace=True, size=len(y))
        mvals.append(np.mean(y[ind]) / np.mean(x[ind]) - 1)

    return np.mean(y) / np.mean(x) - 1, np.std(mvals)


def _fit_m_single(prr):
    g1p, R11p = _get_stuff(prr)

    x = R11p
    y = g1p

    rng = np.random.RandomState(seed=100)
    mvals = []
    for _ in tqdm.trange(1000, leave=False):
        ind = rng.choice(len(y), replace=True, size=len(y))
        mvals.append(np.mean(y[ind]) / np.mean(x[ind]) - 1)

    return np.mean(y) / np.mean(x) - 1, np.std(mvals)


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
files = glob.glob('%s/data*.pkl' % tmpdir)

print('found %d outputs' % len(files))

io = [joblib.delayed(_func)(fname) for fname in files]
outputs = joblib.Parallel(
    verbose=10,
    n_jobs=-1,
    pre_dispatch='2*n_jobs',
    max_nbytes=None)(io)

for i, s2n in enumerate([10, 15, 20]):
    _outputs = []
    for o in outputs:
        _outputs.extend(o[i])
    pres, mres = zip(*_outputs)

    pres, mres = _cut(pres, mres)
    mn, msd = _fit_m(pres, mres)

    print('s2n:', s2n)
    print("""\
    # of sims: {n_sims}
    m       : {mn:f} +/- {msd:f}""".format(
        n_sims=len(pres),
        mn=mn,
        msd=msd), flush=True)

    mn, msd = _fit_m_single(pres)

    print("""\
    m single: {mn:f} +/- {msd:f}""".format(
        mn=mn,
        msd=msd), flush=True)
