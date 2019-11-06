CONFIG = {
    'gal_type': 'wldeblend',
    'psf_type': 'wldeblend',
    'shear_scene': False,
    'n_coadd': 10,  # per band
    'scale': 0.263,
    'n_coadd_psf': 1,
    'gal_kws': {
        'survey_name': 'DES',
        'bands': ('r', 'i', 'z')},
    'ngal_factor': 1,
}

N_PATCHES = 100_000_000
