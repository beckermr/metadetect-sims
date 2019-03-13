import glob

CONFIG = {
    'gal_type': 'exp',
    'psf_type': 'piff',
    'shear_scene': True,
    'n_coadd': 30,
    'scale': 0.263,
    'n_coadd_psf': 1,
    'psf_kws': {'filenames': glob.glob('piffs/*')}
}
