CONFIG = {
    'gal_type': 'exp',
    'psf_type': 'gauss',
    'shear_scene': True,
    'n_coadd': 10,
    'scale': 0.263,
    'n_bands': 3,
    'position_angle_range': (0, 360),
    'scale_frac_std': 0.03,
    'wcs_shear_std': 0.03,
    'wcs_dither_range': (-0.5, 0.5),
    'psf_kws': {'fwhm_frac_std': 0.1, 'shear_std': 0.03}
}

DO_END2END_SIM = True
