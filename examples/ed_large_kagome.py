"""Single-node ED of the 24-site kagome cluster (Sz=0: 2.7M states).

Uses the multithreaded CSR or the matrix-free Numba kernel automatically
(chosen from available memory).  Run with QSLAB_NUM_THREADS=<cores>.
"""
import time
import numpy as np
import qslab
from qslab import lattice, models, ED

print(qslab.parallel_info())
lat = lattice.kagome(2, 4)                     # 24 sites
H = models.kagome_heisenberg(lat)
t = time.time()
ed = ED(H, lat.N)                              # method chosen automatically
w, v = ed.diagonalize(k=4)
print(ed, f"{time.time() - t:.0f}s")
print("E/N =", w / lat.N)
ed_t = ED(H, lat.N, n_up=lat.N // 2 + 1)
wt, _ = ed_t.diagonalize(k=2)
print("spin gap:", wt[0] - w[0])
