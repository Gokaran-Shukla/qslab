# qslab: Quantum Spin Liquid Laboratory

A Python toolkit for simulating quantum spin liquids with complementary many-body
methods. It ships with eight executed tutorial notebooks and quizzes for every
model, and it runs on a laptop, on all cores of one node, across many nodes
(MPI, optional) or on a GPU (CuPy, optional).

```python
import qslab
from qslab import lattice, models, ED, KitaevMajorana

lat = lattice.honeycomb(3, 3)
E_ed, psi = ED(models.kitaev(lat), lat.N).ground_state()   # brute force
E_mf = KitaevMajorana(lat).ground_state_energy()           # exact Majorana solution
print(E_ed, E_mf)        # -14.2915026  -14.2915026
```

## Models

| model | builder | physics |
|---|---|---|
| Kitaev honeycomb (+κ three-spin term, fields) | `models.kitaev` | exactly solvable Z₂ QSL; gapless B phase, gapped A phases, chiral Ising phase |
| J-K-Γ-Γ′-J₃ (α-RuCl₃, Na₂IrO₃) | `models.kitaev_heisenberg_gamma` | Kitaev materials, zigzag order, field-induced phases |
| Toric code (+fields) | `models.toric_code` | Z₂ topological order, anyons, γ = ln 2, confinement |
| Kagome J₁-J₂-J₃ + scalar chirality | `models.kagome_heisenberg` | Z₂ / U(1) Dirac / chiral spin liquids |
| Triangular J₁-J₂ | `models.j1j2_triangular` | 120° order → spin liquid (0.07 ≲ J₂ ≲ 0.15) → stripe |
| Square J₁-J₂ | `models.j1j2_square` | Néel → debated QSL/VBS → stripe; benchmarks |
| Generic Heisenberg/XXZ/DM/field, any Pauli-string Hamiltonian | `models.heisenberg`, `qslab.Operator` | build your own |

Lattices: chain, square, triangular, honeycomb and kagome, as tori, cylinders, strips or
tilted supercells (e.g. √12×√12). Each lattice provides neighbour shells, plaquettes,
translations, momenta and k-paths.

## Methods

| module | method | highlights |
|---|---|---|
| `qslab.ed` | exact diagonalisation | S^z sectors, matrix-free or multithreaded CSR, momenta, total spin, symmetry labels, entanglement, S(q), FTLM thermodynamics, S(q, ω) by continued fraction, time evolution |
| `qslab.kitaev` | exact Majorana solution | any gauge or flux sector, **exact projection onto physical states** (validated level by level against ED), vison gap, Chern number, phase diagram, Majorana Monte Carlo at finite T |
| `qslab.stabilizer` | GF(2) stabilizer formalism | exact entanglement of thousands of qubits; Kitaev-Preskill and Levin-Wen give γ = ln 2 |
| `qslab.parton` | Abrikosov-fermion partons | automatic gauge fixing for **any** flux pattern on any cluster, spinon bands, DOS, mean-field S(q, ω), exact Gutzwiller projection, VMC with Sherman-Morrison updates |
| `qslab.dmrg` | two-site DMRG | MPO generated automatically from any Hamiltonian (long-range 2D bonds), real or complex, entanglement entropy and spectrum, correlations, energy variance |
| `qslab.nqs` | neural quantum states | complex RBM + stochastic reconfiguration, exact or Metropolis sampling, Marshall sign |

## Parallel computing

| level | requirement | what runs in parallel |
|---|---|---|
| threads (default) | `numba` (recommended) | ED matrix-vector products (multithreaded CSR or matrix-free kernel with O(1) Lin-table lookup), BLAS in DMRG and VMC |
| process pool (default) | none | `qslab.pmap`: sectors, FTLM random vectors, parameter scans, MC temperatures and chains |
| MPI across nodes (optional) | `mpi4py` | `pmap` over ranks; VMC, NQS and Majorana-MC chains combined with `allreduce` |
| GPU (optional) | `cupy` | ED (cuSPARSE or kernel), Lanczos, DMRG tensors, Majorana diagonalisation |

```bash
QSLAB_NUM_THREADS=64 python my_script.py           # one node
srun -N 4 --ntasks-per-node=8 python my_script.py  # many nodes (MPI)
QSLAB_BACKEND=gpu python my_script.py              # GPU
qslab-info                                         # print the detected environment
```

Ready-made SLURM scripts are in `examples/` (`slurm_single_node.sh`, `slurm_multi_node.sh`, `slurm_gpu.sh`).

## Install

```bash
pip install -e .                 # core: numpy, scipy, matplotlib
pip install -e ".[fast]"         # + numba, threadpoolctl (multithreaded kernels)
pip install -e ".[mpi]"          # + mpi4py (multi-node)
pip install cupy-cuda12x         # optional GPU backend
pip install -e ".[notebooks,test]"
pytest                           # 21 benchmark tests
```

## Validation (all in `tests/`)

| check | result |
|---|---|
| 4×4 square Heisenberg ED | E₀ = −11.228483208 (exact reference) |
| Kitaev: Majorana vs ED, 8 sites, **full spectrum**, with and without κ | every level identical; uniform gauge multiplicity 2^(N−1) |
| Kitaev: 18-site ground state | Majorana −14.29150262 vs ED −14.29150266 |
| Kitaev thermodynamic limit | E/N = −0.78730 (Pauli units) = −0.19682 (spin units) |
| Kitaev vison pair, ⟨σσ⟩, Chern number | 0.265, 0.5249, ±1 |
| Toric code ED (L=3) | 4 × (−18), gap 4 |
| Toric code γ (KP and LW, L = 8…16) | exactly ln 2 |
| π-flux VMC vs exact Gutzwiller projection (4×4) | −0.63769(35) vs −0.637730 |
| Kagome U(1) Dirac SL, VMC, 108 sites | −0.4291(9) (literature −0.42866) |
| DMRG vs ED (triangular cylinder, complex J-K-Γ cylinder, 4×4 torus) | agreement to 10⁻⁸ |
| FTLM vs full diagonalisation | C(T), E(T) agree |

## Tutorials (`tutorials/`, executed notebooks)

1. `01_getting_started_and_parallel`: lattices, operators, ED, benchmarks, threads, processes, MPI and GPU
2. `02_kitaev_honeycomb`: ED, fluxes, exact Majorana solution, cones, phase diagram, visons, chiral phase, Majorana-MC thermodynamics
3. `03_toric_code`: degeneracy, Wilson loops, anyons, topological entanglement entropy, confinement
4. `04_kagome_heisenberg`: ED spectra, S(q), parton ansätze and the Dirac spin liquid, chiral spin liquid, DMRG
5. `05_triangular_j1j2`: √12 cluster scan, DMRG cylinders, Dirac spin liquid partons
6. `06_square_j1j2_benchmarks_nqs`: benchmarks, π flux, RBM with and without the Marshall sign
7. `07_kitaev_materials_JKGamma`: RuCl₃-like parameters, field scans, complex-MPO DMRG, the DFT → model workflow
8. `08_dynamics_thermodynamics`: S(q, ω) continua, spinon continuum, FTLM C(T), S(T), χ(T)

## Quizzes

```python
from qslab import quiz
quiz.topics()            # kitaev, toric_code, kagome, triangular, square_j1j2, kitaev_heisenberg_gamma, methods
quiz.run("kitaev")       # interactive with score
quiz.check("kitaev", 4, -0.7873)
```
or `qslab-quiz kitaev` from the shell. Computational questions are answered by running qslab.

## Conventions

* Spin operators S = σ/2. `models.kitaev(..., units='pauli')` follows Kitaev (H = −Σ K σσ). Spin units use H = Σ K S S.
* Basis integers: bit i = 0 means ↑ and bit i = 1 means ↓.
* Honeycomb bond labels x, y, z are in `bond.kind`; NN bonds are always oriented A → B.

## Scope and limitations

* ED is practical up to about 30 spins with S^z conservation on a large node (about 24 on a laptop). The Hamiltonian application is shared-memory only; it is not distributed over MPI.
* DMRG is a clear reference implementation without U(1)/SU(2) block sparsity. For production cylinders of width ≥ 8 use TeNPy, ITensor or block2. qslab's MPO builder and models are a convenient cross-check.
* The GPU code paths use only standard CuPy/cuSPARSE calls. They are optional and have not been benchmarked on GPU hardware in this release.
* VMC currently supports U(1) (hopping-only) parton ansätze. Z₂ pairing ansätze are covered at mean-field level through the Kitaev solver.

## Key references

Kitaev, Ann. Phys. 321, 2 (2006) · Kitaev & Preskill, PRL 96, 110404 (2006) · Levin & Wen, PRL 96, 110405 (2006) ·
Ran, Hermele, Lee & Wen, PRL 98, 117205 (2007) · Yan, Huse & White, Science 332, 1173 (2011) ·
Nasu, Udagawa & Motome, PRL 113, 197205 (2014) · Rau, Lee & Kee, PRL 112, 077204 (2014) ·
Zhu & White, PRB 92, 041105 (2015) · Carleo & Troyer, Science 355, 602 (2017) ·
Savary & Balents, Rep. Prog. Phys. 80, 016502 (2017) · Jaklič & Prelovšek, Adv. Phys. 49, 1 (2000).

## License

MIT
