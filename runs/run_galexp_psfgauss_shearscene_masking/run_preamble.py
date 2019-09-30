from mdetsims import (
    TEST_METACAL_MOF_CONFIG,
    TEST_METADETECT_CONFIG,
    TEST_METACAL_TRUEDETECT_CONFIG,
    TEST_METACAL_TRUEDETECT_GAUSS_CONFIG,
    TEST_METACAL_SEP_CONFIG,
    TEST_METACAL_SEP_GAUSS_CONFIG)
from mdetsims import End2EndSim, Sim


def get_shear_meas_config():
    """get config info for shear measurement"""
    # simulation and measurement config stuff
    try:
        from config import SWAP12
    except ImportError:
        SWAP12 = False

    try:
        from config import CUT_INTERP
    except ImportError:
        CUT_INTERP = False

    # set the config for the shear meas algorithm
    try:
        from config import EXTRA_MDET_CONFIG
        TEST_METADETECT_CONFIG.update(EXTRA_MDET_CONFIG)
    except ImportError:
        pass

    try:
        from config import DO_METACAL_MOF
    except ImportError:
        DO_METACAL_MOF = False

    try:
        from config import DO_METACAL_TRUEDETECT
    except ImportError:
        DO_METACAL_TRUEDETECT = False

    try:
        from config import DO_METACAL_SEP
    except ImportError:
        DO_METACAL_SEP = False

    try:
        from config import METACAL_GAUSS_FIT
    except ImportError:
        METACAL_GAUSS_FIT = False

    if DO_METACAL_MOF:
        SHEAR_MEAS_CONFIG = TEST_METACAL_MOF_CONFIG
    elif DO_METACAL_TRUEDETECT:
        if METACAL_GAUSS_FIT:
            SHEAR_MEAS_CONFIG = TEST_METACAL_TRUEDETECT_GAUSS_CONFIG
        else:
            SHEAR_MEAS_CONFIG = TEST_METACAL_TRUEDETECT_CONFIG
    elif DO_METACAL_SEP:
        if METACAL_GAUSS_FIT:
            SHEAR_MEAS_CONFIG = TEST_METACAL_SEP_GAUSS_CONFIG
        else:
            SHEAR_MEAS_CONFIG = TEST_METACAL_SEP_CONFIG
    else:
        SHEAR_MEAS_CONFIG = TEST_METADETECT_CONFIG

    try:
        from config import SHEAR_MEAS_CONFIG
    except ImportError:
        pass

    try:
        from config import DO_END2END_SIM
    except ImportError:
        DO_END2END_SIM = False

    if DO_END2END_SIM:
        SIM_CLASS = End2EndSim
    else:
        SIM_CLASS = Sim

    return (
        SWAP12, CUT_INTERP, DO_METACAL_MOF, DO_METACAL_SEP,
        DO_METACAL_TRUEDETECT,
        SHEAR_MEAS_CONFIG, SIM_CLASS)
