SX_CONFIG = {
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
}

MEDS_CONFIG = {
    'min_box_size': 32,
    'max_box_size': 64,

    'box_type': 'iso_radius',

    'rad_min': 4,
    'rad_fac': 2,
    'box_padding': 2,
}

TEST_METADETECT_CONFIG = {
    'weight': {
        'fwhm': 1.2,  # arcsec
    },

    'metacal': {
        'psf': 'fitgauss',
        'types': ['noshear', '1p', '1m', '2p', '2m'],
        'use_noise_image': True,
    },

    'sx': SX_CONFIG,

    'meds': MEDS_CONFIG,

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

    # flags for mask fractions
    'star_flags': 0,
    'tapebump_flags': 0,
    'spline_interp_flags': 0,
    'noise_interp_flags': 0,
    'imperfect_flags': 0,
}

TEST_METACAL_MOF_CONFIG = {
    'sx': SX_CONFIG,
    'meds': MEDS_CONFIG,
    'fofs': {
        'method': 'radius',
        'check_seg': False,

        # name in meds file, or catalog if method is catalog_radius
        'radius_column': 'iso_radius_arcsec',

        # factor to multiply radius this happens before clipping to
        # [min_radius,max_radius] for a low threshold, the isoarea is
        # basically covering all the observed flux, so mult of 1 makes sense
        'radius_mult': 1.0,

        # clip the radius in pixels=sqrt(isoarea_image/pi)
        # 5 pixels is about 3 sigma for a 1'' FWHM gaussian
        # this happens after the radius_mult is applied
        'min_radius_arcsec': 1.0,
        'max_radius_arcsec': 2.0,

        # This is added to the radius. This kind of padding makes sense for
        # radii based on the iso area.  Padding happens after mult and clipping
        'padding_arcsec': 0.5,

        # arcsec
        'extra_psf_fwhm_arcsec': 0.0
    },

    'weight_type': 'uberseg',

    'mof': {
        'model': 'bdf',

        # number of times to try the fit if it fails
        'ntry': 4,

        # for guesses
        'detband': 0,

        'priors': {
            'cen': {
                'type': 'normal2d',
                'sigma': 0.001
            },
            'g': {
                'type': 'ba',
                'sigma': 0.2
            },
            'T': {
                'type': 'flat',
                'pars': [-1.0, 1.e+05]
            },
            'flux': {
                'type': 'flat',
                'pars': [-1000.0, 1.0e+09]
            },
            'fracdev': {
                'type': 'normal',
                'mean': 0.5,
                'sigma': 0.3,
                'bounds': [-3.0, 4.0]
            }
        },

        'psf': {
            'ntry': 4,
            'model': 'em3',
            'em_pars': {
                'maxiter': 2000,
                'tol': 1.0e-4
            }
        }
    },

    'metacal': {
        # check for an edge hit
        'bmask_flags': 2**30,

        'metacal_pars': {
            'psf': 'fitgauss',
            'types': ['noshear', '1p', '1m', '2p', '2m'],
            # the MOF lib does not set the noise images so we cannot do this
            # 'use_noise_image': True,
        },

        'model': 'gauss',

        'max_pars': {
            'ntry': 2,
            'pars': {
                'method': 'lm',
                'lm_pars': {
                    'maxfev': 2000,
                    'xtol': 5.0e-5,
                    'ftol': 5.0e-5,
                }
            }
        },

        'priors': {
            'cen': {
                'type': 'normal2d',
                'sigma': 0.263
            },

            'g': {
                'type': 'ba',
                'sigma': 0.2
            },

            'T': {
                'type': 'two-sided-erf',
                'pars': [-1.0, 0.1, 1.0e+06, 1.0e+05]
            },

            'flux': {
                'type': 'two-sided-erf',
                'pars': [-100.0, 1.0, 1.0e+09, 1.0e+08]
            }
        },

        'psf': {
            'model': 'gauss',
            'ntry': 2,
            'lm_pars': {
                'maxfev': 2000,
                'ftol': 1.0e-5,
                'xtol': 1.0e-5
            }
        }
    }
}

TEST_METACAL_TRUEDETECT_CONFIG = {
    'meds': MEDS_CONFIG,
    'metacal': {
        # check for an edge hit
        'bmask_flags': 2**30,

        'metacal_pars': {
            'psf': 'fitgauss',
            'types': ['noshear', '1p', '1m', '2p', '2m'],
            'use_noise_image': True,
        },

        'model': 'wmom',

        'weight': {
            'fwhm': 1.2
        },

        # 'model': 'gauss',
        #
        # 'max_pars': {
        #     'ntry': 2,
        #     'pars': {
        #         'method': 'lm',
        #         'lm_pars': {
        #             'maxfev': 2000,
        #             'xtol': 5.0e-5,
        #             'ftol': 5.0e-5,
        #         }
        #     }
        # },
        #
        # 'priors': {
        #     'cen': {
        #         'type': 'normal2d',
        #         'sigma': 0.263
        #     },
        #
        #     'g': {
        #         'type': 'ba',
        #         'sigma': 0.2
        #     },
        #
        #     'T': {
        #         'type': 'two-sided-erf',
        #         'pars': [-1.0, 0.1, 1.0e+06, 1.0e+05]
        #     },
        #
        #     'flux': {
        #         'type': 'two-sided-erf',
        #         'pars': [-100.0, 1.0, 1.0e+09, 1.0e+08]
        #     }
        # },

        'psf': {
            'model': 'gauss',
            'ntry': 2,
            'lm_pars': {
                'maxfev': 2000,
                'ftol': 1.0e-5,
                'xtol': 1.0e-5
            }
        }
    }
}
