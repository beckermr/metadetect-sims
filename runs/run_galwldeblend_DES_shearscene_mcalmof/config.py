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

DO_METACAL_MOF = True

N_PATCHES_PER_JOB = 10
N_PATCHES = 500_000
