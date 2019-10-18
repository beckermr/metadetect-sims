import numpy as np
import galsim
import treecorr
import joblib
import pandas as pd

from mdetsims import PowerSpectrumPSF


def _get_psf_variation(func, n):
    start = 0
    end = start + 225
    jobs = []

    fwhm = np.zeros((n, n))
    g1 = np.zeros((n, n))
    g2 = np.zeros((n, n))

    def __func(j, x):
        retvals = []
        for i, y in enumerate(np.linspace(start, end, n)):
            _psf = func(x, y)
            _psf = _psf.drawImage(scale=0.25, nx=17, ny=17)
            mom = galsim.hsm.FindAdaptiveMom(
                _psf,
                precision=1e-4,
                hsmparams=galsim.hsm.HSMParams(max_mom2_iter=1000))
            retvals.append((
                i, j, mom.moments_sigma * 0.25 * 2.355,
                mom.observed_shape.g1, mom.observed_shape.g2))
        return retvals

    for j, x in enumerate(np.linspace(start, end, n)):
        jobs.append(joblib.delayed(__func)(j, x))

    outputs = joblib.Parallel(
        n_jobs=-1, verbose=10,
        max_nbytes=None, backend='loky')(jobs)
    for o in outputs:
        for i, j, _fwhm, _g1, _g2 in o:
            fwhm[i, j] = _fwhm
            g1[i, j] = _g1
            g2[i, j] = _g2

    return fwhm, g1, g2


def _measure_xi(seed, n, n_stack=1):
    rng = np.random.RandomState(seed=seed)

    psfs = [
        PowerSpectrumPSF(
            rng=rng,
            im_width=225,
            buff=225//2,
            scale=0.263,
            median_seeing=0.8)
        for _ in range(n_stack)]

    def _func(x, y):
        if len(psfs) > 1:
            return galsim.Sum([
                psf.getPSF(galsim.PositionD(x=x, y=y))
                for psf in psfs]).withFlux(1.0)
        else:
            return psfs[0].getPSF(galsim.PositionD(x=x, y=y))

    fwhm, g1, g2 = _get_psf_variation(_func, n=n)

    y, x = np.mgrid[:n, :n] * 0.263 * 225 / n
    cat = treecorr.Catalog(
        x=y.ravel(), y=x.ravel(), g1=g1.ravel(), g2=g2.ravel(),
        x_units='arcsec', y_units='arcsec')

    gg = treecorr.GGCorrelation(
        nbins=50, min_sep=1, max_sep=60, bin_slop=0.1,
        sep_units='arcsec')

    gg.process_auto(cat)
    gg.finalize(np.std(g1)**2, np.std(g2)**2)

    return gg.rnom, gg.xip, gg.xim


rng = np.random.RandomState(seed=419)
seeds = rng.randint(1, 2**30, size=100)
sims = [joblib.delayed(_measure_xi)(seed, 225, n_stack=1) for seed in seeds]
outputs = joblib.Parallel(
    verbose=20,
    n_jobs=1,
    pre_dispatch='2*n_jobs',
    max_nbytes=None,
    backend='loky')(sims)

r = np.mean(np.array([o[0] for o in outputs]), axis=0)
xip = np.mean(np.array([o[1] for o in outputs]), axis=0)
xim = np.mean(np.array([o[2] for o in outputs]), axis=0)

df = pd.DataFrame({'r': r, 'xip': xip, 'xim': xim})
df.to_csv('../ps_psf_xi.csv', index=False)
