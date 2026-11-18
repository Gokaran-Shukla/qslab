"""DMRG scan of the triangular J1-J2 model on a width-6 cylinder (single node)."""
import numpy as np
import qslab
from qslab import lattice, models, DMRG

for J2 in [0.0, 0.07, 0.1, 0.125, 0.15]:
    lat = lattice.triangular(12, 6, bc="cylinder")
    d = DMRG(models.j1j2_triangular(lat, J2=J2), lat.N, chi_max=600)
    d.run(n_sweeps=14, chi_schedule=[50, 100, 200, 300, 400, 500, 600] + [600] * 7, verbose=False)
    print(f"J2={J2:.3f}  E/N={d.energy / lat.N:.6f}  S_mid={d.entropies[lat.N // 2 - 1]:.4f}")
