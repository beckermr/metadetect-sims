import copy
from mdetsims.defaults import TEST_METACAL_TRUEDETECT_GAUSS_CONFIG

CONFIG = {
    'gal_type': 'exp',
    'psf_type': 'gauss',
    'shear_scene': True,
    'n_coadd': 30,
    'scale': 0.263,
    'gal_grid': 7,
}

N_PATCHES = 1_000_000

DO_METACAL_TRUEDETECT = True
METACAL_GAUSS_FIT = True

SHEAR_MEAS_CONFIG = copy.deepcopy(TEST_METACAL_TRUEDETECT_GAUSS_CONFIG)
SHEAR_MEAS_CONFIG['metacal']['keep_all_fits'] = True
