import numpy as np

from numba import njit


@njit(fastmath=True)
def sinc_pade(x):
    """A pseudo-Pade approximation to sinc.

    When used in lanczos interpolation, this approximation to the sinc
    function has error in the kernel of at most ~1.03019e-05 and is exactly
    zero at x = 1, 2, 3 and exactly 1 at x = 0.
    """
    x = np.abs(x)
    num = (  # noqa
        -0.166666666666666 + x * (
        -0.289176685343373 + x * (
        -0.109757669089546 + x * (
        0.0350931080575596 + x * (
        0.0229947584643336 + x * (
        -0.00089363958201935 + x * (
        -0.00162722192965722 + x * (
        -3.00689146075626e-05 + x * (
        5.13864469774294e-05 + x * (
        1.23561563382214e-06 + x * (
        -6.37392253619041e-07))))))))))) * (x-1) * (x-2) * (x-3)
    den = (  # noqa
        1 + x * (
        -0.0982732212730775 + x * (
        0.122536542608403 + x * (
        -0.0111525324680647 + x * (
        0.00724707512833019 + x * (
        -0.000584774445653404 + x * (
        0.000262528048296579 + x * (
        -1.71596576734417e-05 + x * (
        5.91945411660804e-06 + x * (
        -2.44174818579491e-07 + x * (
        6.74473938075399e-08)))))))))))
    return num / den
