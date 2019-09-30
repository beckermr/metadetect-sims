import copy
from mdetsims.defaults import TEST_METACAL_TRUEDETECT_CONFIG

CONFIG = {
    'gal_type': 'wldeblend',
    'psf_type': 'wldeblend',
    'shear_scene': True,
    'n_coadd': 10,  # per band
    'scale': 0.263,
    'n_coadd_psf': 1,
    'gal_kws': {
        'survey_name': 'DES',
        'bands': ('r', 'i', 'z')},
}

DO_METACAL_TRUEDETECT = True

SHEAR_MEAS_CONFIG = copy.deepcopy(TEST_METACAL_TRUEDETECT_CONFIG)
SHEAR_MEAS_CONFIG['metacal']['keep_all_fits'] = True

N_PATCHES_PER_JOB = 50  # these ones take a while
