CONFIG = {
    'gal_type': 'wldeblend',
    'psf_type': 'ps',
    'shear_scene': True,
    'n_coadd': 10,
    'scale': 0.263,
    'n_bands': 3,
    'position_angle_range': (0, 360),
    'scale_frac_std': 0.03,
    'wcs_shear_std': 0.03,
    'wcs_dither_range': (-0.5, 0.5),
    'mask_and_interp': True,
    'interpolation_type': 'cubic',
    'gal_kws': {
        'survey_name': 'DES',
        'bands': ('r', 'i', 'z')},
}

DO_END2END_SIM = True
N_PATCHES_PER_JOB = 50
