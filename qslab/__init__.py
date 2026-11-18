"""
qslab: a toolkit for simulating quantum spin liquids
====================================================

Models    Kitaev honeycomb (+kappa, fields), J-K-Gamma-Gamma'-J3, toric code,
          J1-J2 square, J1-J2 triangular, kagome J1-J2-J3-chiral,
          generic Heisenberg/XXZ/DM, plus any custom Pauli-string Hamiltonian
Methods   exact diagonalisation, exact Majorana solution, stabilizer TEE,
          partons + Gutzwiller VMC, DMRG, neural quantum states,
          FTLM and Majorana Monte Carlo thermodynamics, S(q, w)
Parallel  threads + process pool on one node (default); optional MPI over
          several nodes; optional CuPy GPU backend

Quick start::

    import qslab
    from qslab import lattice, models, ED
    lat = lattice.kagome(2, 2)
    ed = ED(models.kagome_heisenberg(lat), lat.N)
    E0, psi = ed.ground_state()

Tutorials and quizzes: ``qslab.quiz.run('kitaev')`` or ``python -m qslab.quiz``.
"""
__version__ = "0.1.0"

from . import parallel
from .parallel import (set_num_threads, get_num_threads, set_backend, get_backend,
                       pmap, parallel_info, mpi_info, is_root, rank_print)
from . import lattice, operators, models
from .operators import (Operator, sigma, spin, sdot, exchange, scalar_chirality,
                        pauli_string, total_spin, total_sz)
from .ed import ED, SpinBasis, ftlm_thermodynamics, spectrum_all_sectors
from . import kitaev, stabilizer, parton, dmrg, nqs
from .kitaev import KitaevMajorana, majorana_monte_carlo
from .dmrg import DMRG
from .parton import SpinonAnsatz, GutzwillerVMC, projected_state
from .nqs import NQS

__all__ = [
    "lattice", "operators", "models", "kitaev", "stabilizer", "parton", "dmrg", "nqs",
    "ED", "SpinBasis", "DMRG", "KitaevMajorana", "SpinonAnsatz", "GutzwillerVMC", "NQS",
    "Operator", "sigma", "spin", "sdot", "exchange", "scalar_chirality", "pauli_string",
    "set_num_threads", "set_backend", "pmap", "parallel_info", "mpi_info",
]
