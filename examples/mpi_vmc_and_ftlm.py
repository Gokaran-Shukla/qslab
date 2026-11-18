"""Multi-node example (run with mpirun/srun).  Every part is MPI-aware:

* Gutzwiller VMC: one or more Markov chains per rank, merged statistics
* FTLM: random vectors and Sz sectors distributed over ranks
* pmap parameter scan over J2 (round-robin over ranks)
* Majorana Monte Carlo: temperatures distributed over ranks
"""
import numpy as np
import qslab
from qslab import lattice, models, ED
from qslab.parton import SpinonAnsatz, GutzwillerVMC
from qslab.ed import ftlm_thermodynamics
from qslab.kitaev import majorana_monte_carlo

rank, size = qslab.mpi_info()
qslab.rank_print(f"running on {size} MPI ranks", qslab.parallel_info())

# 1. VMC of the kagome Dirac spin liquid on 432 sites
lat = lattice.kagome(12, 12)
ans = SpinonAnsatz.from_flux(lat, {"up": 0, "down": 0, "hexagon": np.pi})
res = GutzwillerVMC(ans).run(n_sweeps=2000, n_therm=200, n_chains=1)
qslab.rank_print("DSL E/N =", res["E_per_site"], "+-", res["err_per_site"], "chains:", res["chains"])

# 2. FTLM thermodynamics of the 24-site kagome cluster
T = np.geomspace(0.02, 5, 60)
th = ftlm_thermodynamics(models.kagome_heisenberg(lattice.kagome(2, 4)), 24, T, R=8, M=100)
if qslab.is_root():
    np.savetxt("kagome24_ftlm.dat", np.column_stack([T, th["C"], th["S"], th["chi"]]),
               header="T C S chi")

# 3. J2 scan of the triangular model on the sqrt12 cluster
def point(J2):
    lat = lattice.triangular(1, supercell=[[2, 2], [-2, 4]])
    w, _ = ED(models.j1j2_triangular(lat, J2=J2), lat.N).diagonalize(k=2)
    return w[0] / lat.N, w[1] - w[0]
scan = qslab.pmap(point, np.linspace(0, 0.3, 31))
qslab.rank_print("scan done:", scan[:3])

# 4. Kitaev thermodynamics
mc = majorana_monte_carlo(L=8, temperatures=np.geomspace(0.003, 5, 4 * size), n_therm=500, n_meas=2000)
if qslab.is_root():
    np.savetxt("kitaev_mc.dat", np.column_stack([mc["T"], mc["C"], mc["S"], mc["W"]]), header="T C S Wp")
