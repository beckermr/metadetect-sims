CONFIG = {
    'gal_type': 'exp',
    'psf_type': 'gauss',
    'shear_scene': True,
    'n_coadd': 30,
    'scale': 0.263,
    'mask_and_interp': True,
    'bad_columns_kws': dict(
        widths=(1, 2),
        p=(0.8, 0.2),
        min_length_frac=(1, 1),
        max_length_frac=(1, 1),
        gap_prob=(0.30, 0.30),
        min_gap_frac=(0.1, 0.1),
        max_gap_frac=(0.3, 0.3)
    )
}
