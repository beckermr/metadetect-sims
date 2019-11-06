import numpy as np
import pytest

from ..lanczos import sinc


def test_sinc():
    al = 3
    xv = np.linspace(-al, al, 10000)

    eps = np.max(np.abs(np.array(
        [sinc(x) * sinc(x/al) - np.sinc(x) * np.sinc(x/al)
         for x in xv])))
    assert eps < 1e-32


@pytest.mark.parametrize('v', [-1e-5, -1e-12, 0, 1e-12, 1e-5])
def test_sinc_small(v):
    assert np.allclose(np.sinc(v), sinc(v))
