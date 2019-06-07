import time

import numpy as np
from mdetsims.lanczos import lanczos_resample_three


dim = 500

rng = np.random.RandomState(seed=10)
im1 = rng.normal(size=(dim, dim))
im2 = rng.normal(size=(dim, dim))
im3 = rng.normal(size=(dim, dim))

row = rng.uniform(size=10000, low=0, high=dim-1)
col = rng.uniform(size=10000, low=0, high=dim-1)

val1, val2, val3, _ = lanczos_resample_three(im1, im2, im3, row, col, a=3)

t0 = time.time()
for _ in range(10):
    val1, val2, val3, _ = lanczos_resample_three(im1, im2, im3, row, col, a=3)

t0 = time.time() - t0

print('time:', t0/10)
