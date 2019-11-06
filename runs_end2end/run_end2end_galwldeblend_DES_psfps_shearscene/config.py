CONFIG = {
    'gal_type': 'wldeblend',
    'psf_type': 'ps',
    'shear_scene': True,
    'n_coadd': 10,
    'scale': 0.263,
    'n_bands': 3,
    'gal_kws': {
        'survey_name': 'DES',
        'bands': ('r', 'i', 'z')},
}

DO_END2END_SIM = True
N_PATCHES_PER_JOB = 50
