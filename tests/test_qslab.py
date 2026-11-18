"""Benchmark tests: every solver is checked against an exact or literature result."""
import itertools

import numpy as np
import pytest

import qslab
from qslab import ED, lattice, models as M, sdot
from qslab.operators import Operator, sigma, spin, CompiledOperator
from qslab.ed import SpinBasis, thermodynamics_from_spectrum, ftlm_thermodynamics


# ----------------------------------------------------------------- operators
def test_pauli_algebra():
    x, y, z = sigma(0, "x"), sigma(0, "y"), sigma(0, "z")
    assert (x * y - 1j * z).terms == {}
    assert (x * x).terms == {(): 1}
    assert sdot(0, 1).conserves_sz()
    assert not (sigma(0, "x")).conserves_sz()


def test_operator_matrix_vs_kron():
    X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1])
    I = np.eye(2)
    op = 0.3 * sigma(0, "x") * sigma(1, "y") + 0.7 * sigma(1, "z") - 0.2 * sigma(0, "y")
    ref = 0.3 * np.kron(Y, X) + 0.7 * np.kron(Z, I) - 0.2 * np.kron(I, Y)  # site 1 = high bit
    assert np.allclose(op.to_matrix(2), ref)


@pytest.mark.parametrize("method", ["numpy", "numba"])
def test_kernels_match_sparse(method):
    lat = lattice.honeycomb(2, 2)
    H = M.kitaev_heisenberg_gamma(lat, J=-0.1, K=-1, Gamma=0.4, Gamma_p=-0.05, h=(0.1, 0.2, 0.3))
    b = SpinBasis(lat.N)
    v = np.random.default_rng(0).standard_normal(b.dim) + 0j
    c = CompiledOperator(H, lat.N)
    assert np.allclose(c.apply(b, v, method=method), c.to_sparse(b) @ v)


# ------------------------------------------------------------------- ED
def test_square_4x4_heisenberg():
    lat = lattice.square(4)
    E0, psi = ED(M.heisenberg(lat), lat.N).ground_state()
    assert abs(E0 - (-11.228483208428)) < 1e-8


def test_chain_and_total_spin():
    lat, H = M.xxz_chain(12)
    ed = ED(H, 12)
    E0, psi = ed.ground_state()
    assert abs(E0 / 12 - (-0.4489492)) < 1e-6      # L=12 PBC
    assert abs(ed.total_spin(psi)) < 1e-6


def test_momentum_quantum_numbers():
    lat = lattice.square(4)
    ed = ED(M.heisenberg(lat), lat.N)
    ed.diagonalize(k=4)
    (E, k), = ed.momenta(lat)[:1]
    assert np.allclose(k, (0, 0))


def test_ftlm_vs_exact():
    lat, H = M.xxz_chain(10)
    T = np.array([0.3, 0.6, 1.2])
    ex = thermodynamics_from_spectrum(np.linalg.eigvalsh(H.to_matrix(10).real), T, 10)
    ft = ftlm_thermodynamics(H, 10, T, R=40, M=80)
    assert np.allclose(ft["E"], ex["E"], atol=5e-3)
    assert np.allclose(ft["C"], ex["C"], atol=3e-2)


def test_entanglement_bell_like():
    lat, H = M.xxz_chain(2, bc="open")
    ed = ED(H, 2)
    E, psi = ed.ground_state()
    assert abs(ed.entanglement_entropy(psi, [0]) - np.log(2)) < 1e-10


# --------------------------------------------------------------- Kitaev
@pytest.mark.parametrize("K,kappa", [((1, 1, 1), 0.0), ((1, 0.7, 1.2), 0.2)])
def test_kitaev_majorana_full_spectrum(K, kappa):
    from qslab.kitaev import KitaevMajorana
    lat = lattice.honeycomb(2, 2)
    km = KitaevMajorana(lat, K, kappa)
    cnt = {}
    for bits in itertools.product([1, -1], repeat=km.nb):
        u = np.array(bits, float)
        eps = km.spectrum(u)
        P = km.is_physical(u)
        E0 = -0.5 * eps.sum()
        for occ in itertools.product([0, 1], repeat=len(eps)):
            if (sum(occ) % 2 == 0) == P:
                e = round(E0 + np.dot(occ, eps), 7)
                cnt[e] = cnt.get(e, 0) + 1
    w = np.linalg.eigvalsh(M.kitaev(lat, K, kappa=kappa).to_matrix(lat.N))
    edc = {}
    for e in w:
        edc[round(e, 7)] = edc.get(round(e, 7), 0) + 1
    assert set(cnt) == set(edc)
    assert len({cnt[k] / edc[k] for k in edc}) == 1          # uniform gauge redundancy


def test_kitaev_18_and_tl():
    from qslab.kitaev import KitaevMajorana, kitaev_energy_tl
    lat = lattice.honeycomb(3, 3)
    km = KitaevMajorana(lat)
    E_ed, _ = ED(M.kitaev(lat), lat.N).ground_state()
    assert abs(km.ground_state_energy() - E_ed) < 1e-7
    assert abs(kitaev_energy_tl(nk=120) - (-0.78730)) < 2e-4


def test_kitaev_fluxes_conserved():
    lat = lattice.honeycomb(2, 3)
    H = M.kitaev(lat)
    for W in M.kitaev_plaquette_operators(lat):
        assert H.commutes_with(W)


def test_chern():
    from qslab.kitaev import chern_number_fhs, kitaev_bloch
    B = lattice.honeycomb(1, 1).reciprocal()
    c = chern_number_fhs(lambda k: kitaev_bloch(k, (1, 1, 1), 0.1), B, nk=24)
    assert abs(abs(c) - 1) < 1e-6


# ----------------------------------------------------------- toric code
def test_toric_code_ed_and_tee():
    from qslab.stabilizer import toric_code_ground_state, kitaev_preskill_regions, \
        levin_wen_regions, topological_entropy
    tc, H = M.toric_code(3)
    ed = ED(H, tc.N)
    w, _ = ed.diagonalize(k=5)
    assert np.allclose(w[:4], -18) and abs(w[4] + 14) < 1e-8
    tc, st = toric_code_ground_state(12)
    assert abs(topological_entropy(st, *kitaev_preskill_regions(tc)) - np.log(2)) < 1e-12
    assert abs(topological_entropy(st, *levin_wen_regions(tc), scheme="LW") - np.log(2)) < 1e-12


# --------------------------------------------------------------- partons
def test_projection_and_vmc():
    from qslab.parton import SpinonAnsatz, projected_state, GutzwillerVMC
    lat = lattice.square(4)
    ans = SpinonAnsatz.from_flux(lat, {"square": np.pi}, twist=(0, np.pi))
    basis, psi = projected_state(ans)
    ed = ED(M.heisenberg(lat), lat.N)
    E_exact = np.vdot(psi, ed.matvec(psi)).real / lat.N
    r = GutzwillerVMC(ans).run(n_sweeps=1500, n_therm=100, seed=3)
    assert abs(r["E_per_site"] - E_exact) < 5 * r["err_per_site"] + 2e-3


def test_flux_solver_kagome():
    from qslab.parton import SpinonAnsatz
    lat = lattice.kagome(4, 4)
    ans = SpinonAnsatz.from_flux(lat, {"up": 0, "down": 0, "hexagon": np.pi})
    fl = ans.flux_through(["up", "down", "hexagon"])
    assert np.allclose(np.cos(fl["hexagon"]), -1) and np.allclose(np.cos(fl["up"]), 1)


# ------------------------------------------------------------------ DMRG
def test_dmrg_vs_ed():
    from qslab.dmrg import DMRG
    lat = lattice.triangular(4, 3, bc="cylinder")
    H = M.j1j2_triangular(lat, J2=0.1)
    E_ed, _ = ED(H, lat.N).ground_state()
    d = DMRG(H, lat.N, chi_max=128)
    d.run(10, verbose=False)
    assert abs(d.energy - E_ed) < 1e-6


def test_dmrg_complex_kgamma():
    from qslab.dmrg import DMRG
    lat = lattice.honeycomb(3, 2, bc="cylinder")
    H = M.kitaev_heisenberg_gamma(lat, J=-0.1, K=-1, Gamma=0.5)
    E_ed, _ = ED(H, lat.N).ground_state()
    d = DMRG(H, lat.N, chi_max=64)
    d.run(8, verbose=False)
    assert abs(d.energy - E_ed) < 1e-7


# ------------------------------------------------------------------- NQS
def test_nqs_small():
    from qslab.nqs import NQS
    lat, H = M.xxz_chain(8)
    E0, _ = ED(H, 8).ground_state()
    q = NQS(H, 8, alpha=2, marshall=lat.sublattice_sign())
    E = q.optimize(150, lr=0.05, verbose=0)
    assert (E - E0) / abs(E0) < 1e-2


# -------------------------------------------------------------- parallel
def _sq(x):
    return x * x


def test_pmap_backends():
    for b in ("serial", "thread", "process"):
        assert qslab.pmap(_sq, range(5), backend=b, n_jobs=2) == [0, 1, 4, 9, 16]
    assert qslab.pmap(lambda x: x + 1, [1, 2], backend="process", n_jobs=2) == [2, 3]
