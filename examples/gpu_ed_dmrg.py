"""Optional GPU backend (CuPy).  Falls back with a clear error if no GPU."""
import time
import numpy as np
import qslab
from qslab import lattice, models, ED, DMRG
from qslab.kitaev import KitaevMajorana

qslab.set_backend("gpu")
print(qslab.parallel_info())

lat = lattice.kagome(2, 4)
t = time.time()
ed = ED(models.kagome_heisenberg(lat), lat.N, method="gpu")      # cuSPARSE matvec
print("GPU ED E/N:", ed.ground_state()[0] / lat.N, f"{time.time() - t:.0f}s")

lat = lattice.triangular(12, 6, bc="cylinder")
d = DMRG(models.j1j2_triangular(lat, J2=0.1), lat.N, chi_max=1024, backend="gpu")
d.run(n_sweeps=12)

km = KitaevMajorana(lattice.honeycomb(60, 60), backend="gpu")    # 7200 Majoranas
print("Kitaev E/N:", km.matter_ground_energy() / km.N)
