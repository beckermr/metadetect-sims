CONFIG = {
    'gal_type': 'wldeblend',
    'psf_type': 'wldeblend',
    'shear_scene': True,
    'n_coadd': 1,  # ignored for this sim
    'scale': 0.2,
    'dim': 300,
    'n_coadd_psf': 1,
    'gal_kws': {
        'survey_name': 'LSST',
        'bands': ('r', 'i', 'z')},
}

DO_METACAL_MOF = True

N_PATCHES_PER_JOB = 10
N_PATCHES = 50000
