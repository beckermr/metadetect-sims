import glob
import pickle
import joblib


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
jobs = glob.glob('*.desc')

if len(files) == len(jobs):
    print('found %d outputs' % len(files))

    io = [joblib.delayed(_func)(fname) for fname in files]
    outputs = joblib.Parallel(
        verbose=10,
        n_jobs=-1,
        pre_dispatch='2*n_jobs',
        max_nbytes=None)(io)

    with open('outputs/final.pkl', 'wb') as fp:
        pickle.dump(outputs, fp)
