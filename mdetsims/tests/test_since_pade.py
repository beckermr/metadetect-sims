import numpy as np

from ..lanczos import sinc_pade


def test_sinc_pade():
    al = 3
    x = np.linspace(-al, al, 10000)

    eps = np.max(np.abs(
        sinc_pade(x) * sinc_pade(x/al) - np.sinc(x) * np.sinc(x/al)))
    assert eps < 7e-8
