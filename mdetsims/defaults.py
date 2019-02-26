TEST_METADETECT_CONFIG = {
    'weight': {
        'fwhm': 1.2,  # arcsec
    },

    'metacal': {
        'psf': 'fitgauss',
        'types': ['noshear', '1p', '1m', '2p', '2m'],
        'use_noise_image': True,
    },

    'sx': {
        # in sky sigma
        # DETECT_THRESH
        'detect_thresh': 0.8,

        # Minimum contrast parameter for deblending
        # DEBLEND_MINCONT
        'deblend_cont': 0.00001,

        # minimum number of pixels above threshold
        # DETECT_MINAREA: 6
        'minarea': 4,

        'filter_type': 'conv',

        # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
        'filter_kernel':  [
            [0.004963, 0.021388, 0.051328, 0.068707,
             0.051328, 0.021388, 0.004963],
            [0.021388, 0.092163, 0.221178, 0.296069,
             0.221178, 0.092163, 0.021388],
            [0.051328, 0.221178, 0.530797, 0.710525,
             0.530797, 0.221178, 0.051328],
            [0.068707, 0.296069, 0.710525, 0.951108,
             0.710525, 0.296069, 0.068707],
            [0.051328, 0.221178, 0.530797, 0.710525,
             0.530797, 0.221178, 0.051328],
            [0.021388, 0.092163, 0.221178, 0.296069,
             0.221178, 0.092163, 0.021388],
            [0.004963, 0.021388, 0.051328, 0.068707,
             0.051328, 0.021388, 0.004963],
        ]
    },

    'meds': {
        'min_box_size': 32,
        'max_box_size': 256,

        'box_type': 'iso_radius',

        'rad_min': 4,
        'rad_fac': 2,
        'box_padding': 2,
    },

    # needed for PSF symmetrization
    'psf': {
        'model': 'gauss',

        'ntry': 2,

        'lm_pars': {
            'maxfev': 2000,
            'ftol': 1.0e-5,
            'xtol': 1.0e-5,
        }
    },

    # check for an edge hit
    'bmask_flags': 2**30,
}
