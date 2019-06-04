CONFIG = {
    'gal_type': 'exp',
    'psf_type': 'gauss',
    'shear_scene': True,
    'n_coadd': 30,
    'scale': 0.263,
    'mask_and_interp': True
}

CUT_INTERP = True

EXTRA_MDET_CONFIG = {
    # mask bigger regions around the center
    'ormask_region': 2,  # 5x5 pixel region around object center
}
