from ..real_psf import get_good_fft_sizes


def test_get_good_fft_sizes():
    sze = [31]
    assert get_good_fft_sizes(sze)[0] == 32

    sze = [32]
    assert get_good_fft_sizes(sze)[0] == 32

    sze = [33]
    assert get_good_fft_sizes(sze)[0] == 64

    sze = [3185479423]
    assert get_good_fft_sizes(sze)[0] == 8192
