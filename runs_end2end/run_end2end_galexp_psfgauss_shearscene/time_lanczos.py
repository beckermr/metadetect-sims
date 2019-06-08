import time

import numpy as np


if True:
    from mdetsims._lanczos import lanczos_one

    dim = 500

    rng = np.random.RandomState(seed=10)
    im1 = rng.normal(size=(dim, dim))

    row = rng.uniform(size=10000, low=0, high=dim-1)
    col = rng.uniform(size=10000, low=0, high=dim-1)

    val1, _ = lanczos_one(im1, row, col, 3)

    t0 = time.time()
    for _ in range(10):
        val1, _ = lanczos_one(im1, row, col, 3)
    t0 = time.time() - t0

    print('time:', t0/10)

else:
    from mdetsims.lanczos import lanczos_resample_one

    dim = 500

    rng = np.random.RandomState(seed=10)
    im1 = rng.normal(size=(dim, dim))

    row = rng.uniform(size=10000, low=0, high=dim-1)
    col = rng.uniform(size=10000, low=0, high=dim-1)

    val1, _ = lanczos_resample_one(im1, row, col, a=3)

    t0 = time.time()
    for _ in range(10):
        val1, _ = lanczos_resample_one(
            im1, row, col, a=3)
    t0 = time.time() - t0

    print('time:', t0/10)
