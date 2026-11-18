"""
Exact solution of the Kitaev honeycomb model via Majorana fermions.

    H = - sum_<jk>_g K_g sigma^g_j sigma^g_k  - kappa sum sigma^a_j sigma^g_k sigma^b_l

Kitaev's representation sigma^a_j = i b^a_j c_j turns H, for a fixed
configuration of the conserved Z2 gauge field u_jk = i b^a_j b^a_k, into free
Majorana fermions::

    H_u = (i/4) sum_jk A_jk c_j c_k,   A_jk = 2 K_g u_jk  (NN, j in A)
                                       A_jl = -2 kappa eps_abg u_jk u_kl  (NNN)

This module provides:

* exact spectra in any gauge / flux / topological sector, on any periodic
  honeycomb cluster (rectangular or tilted);
* exact projection onto the physical subspace (prod_j D_j = 1).  The sign
  bookkeeping is done numerically: Pfaffian parity of the matter ground
  state plus the permutation sign of all Majoranas, so there is no
  lattice-specific formula to get wrong.  The result agrees with ED to
  machine precision;
* fluxes W_p, Wilson loops, vison (flux-pair) gaps;
* the Majorana band structure, Chern number (Fukui-Hatsugai) and phase
  diagram (gapless B phase / gapped A phases / chiral phase for kappa != 0);
* spin-spin correlations;
* finite-temperature Majorana Monte Carlo over gauge fields (Nasu, Udagawa
  and Motome, PRL 113, 197205 (2014)), parallel over temperatures and MPI
  ranks.  It gives the two-peak specific heat and half-ln2 entropy plateau.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .lattice import Lattice, honeycomb
from .models import kitaev_kappa_triples
from .parallel import pmap, xp_module, to_numpy

__all__ = ["pfaffian", "KitaevMajorana", "kitaev_phase", "kitaev_bloch",
           "chern_number_fhs", "majorana_monte_carlo", "kitaev_energy_tl"]


# --------------------------------------------------------------------------- #
def pfaffian(A: np.ndarray) -> Tuple[float, float]:
    """Pfaffian of a real/complex skew-symmetric matrix.

    Returns ``(sign_or_phase, log|Pf|)``.  Parlett-Reid with pivoting.
    """
    A = np.array(A, dtype=complex if np.iscomplexobj(A) else float, copy=True)
    n = A.shape[0]
    if n % 2:
        return 0.0, -np.inf
    sgn = 1.0 + 0j if np.iscomplexobj(A) else 1.0
    logabs = 0.0
    for k in range(0, n - 1, 2):
        kp = k + 1 + int(np.argmax(np.abs(A[k + 1:, k])))
        if kp != k + 1:
            A[[k + 1, kp], :] = A[[kp, k + 1], :]
            A[:, [k + 1, kp]] = A[:, [kp, k + 1]]
            sgn = -sgn
        piv = A[k, k + 1]
        if abs(piv) < 1e-300:
            return 0.0, -np.inf
        sgn *= piv / abs(piv)
        logabs += np.log(abs(piv))
        if k + 2 < n:
            tau = A[k, k + 2:] / piv
            v = A[k + 2:, k + 1].copy()
            A[k + 2:, k + 2:] += np.outer(tau, v) - np.outer(v, tau)
    return sgn, logabs


def _perm_parity(seq_from: List, seq_to: List) -> int:
    """Sign of the permutation taking seq_from to seq_to (Majorana reordering)."""
    pos = {x: n for n, x in enumerate(seq_from)}
    p = np.array([pos[x] for x in seq_to])
    seen = np.zeros(len(p), bool)
    parity = 0
    for i in range(len(p)):
        if not seen[i]:
            j, L = i, 0
            while not seen[j]:
                seen[j] = True
                j = p[j]
                L += 1
            parity += L - 1
    return -1 if parity % 2 else 1


# --------------------------------------------------------------------------- #
class KitaevMajorana:
    """Free-Majorana solver of the Kitaev model on a periodic honeycomb cluster.

    Parameters
    ----------
    lat : Lattice
        Periodic honeycomb cluster (``honeycomb(L1, L2)`` or a supercell),
        L >= 2 in each direction.
    K : (Kx, Ky, Kz)
        Couplings in Pauli units (H = -sum K sigma sigma).
    kappa : float
        Time-reversal-breaking three-spin coupling (opens a gap, Chern +-1).
    backend : 'cpu' | 'gpu'
        Diagonalisation on CuPy for very large clusters.

    Example
    -------
    >>> km = KitaevMajorana(honeycomb(12, 12))
    >>> km.ground_state_energy() / km.N      # -> -0.7873 (thermodynamic limit)
    """

    def __init__(self, lat: Lattice, K: Sequence[float] = (1.0, 1.0, 1.0),
                 kappa: float = 0.0, backend: str = "cpu"):
        if lat.name != "honeycomb" or not all(lat.periodic):
            raise ValueError("need a periodic honeycomb cluster")
        self.lat = lat
        self.N = lat.N
        self.K = dict(zip("xyz", map(float, K)))
        self.kappa = float(kappa)
        self.backend = backend
        self.bonds = lat.bonds(1)            # i always on sublattice A
        self.nb = len(self.bonds)
        if self.nb != 3 * self.N // 2:
            raise ValueError("cluster too small (bonds collide)")
        self._bond_of = {}
        for n, b in enumerate(self.bonds):
            self._bond_of[(b.i, b.j)] = n
            self._bond_of[(b.j, b.i)] = n
        self.u = np.ones(self.nb)
        self.triples = kitaev_kappa_triples(lat) if kappa else []
        # bond kinds per site
        self._site_bond = {}
        for n, b in enumerate(self.bonds):
            self._site_bond[(b.i, b.kind)] = n
            self._site_bond[(b.j, b.kind)] = n
        self._setup_parity_bookkeeping()

    # ---------------------------------------------------------------- #
    def _setup_parity_bookkeeping(self):
        """Pairing of c-Majoranas along z-bonds and the global reordering sign."""
        zb = [b for b in self.bonds if b.kind == "z"]
        self.pair_order = []
        for b in zb:
            self.pair_order += [b.i, b.j]
        init = [(j, t) for j in range(self.N) for t in ("x", "y", "z", "c")]
        target = []
        for b in self.bonds:
            target += [(b.i, b.kind), (b.j, b.kind)]
        for b in zb:
            target += [(b.i, "c"), (b.j, "c")]
        self._s_perm = _perm_parity(init, target)
        # prod_j D_j = s_perm * (-1)^{N/2} * prod u * P_matter
        self._const = self._s_perm * (-1) ** (self.N // 2)

    # ---------------------------------------------------------------- #
    def u_directed(self, j: int, k: int, u: np.ndarray = None) -> float:
        """u_jk = i b_j b_k for the bond (j,k) in the given order."""
        u = self.u if u is None else u
        n = self._bond_of[(j, k)]
        return u[n] if self.bonds[n].i == j else -u[n]

    def A_matrix(self, u: np.ndarray = None) -> np.ndarray:
        """Real antisymmetric hopping matrix A (H = i/4 c^T A c)."""
        u = self.u if u is None else np.asarray(u)
        A = np.zeros((self.N, self.N))
        for n, b in enumerate(self.bonds):
            v = 2.0 * self.K[b.kind] * u[n]
            A[b.i, b.j] += v
            A[b.j, b.i] -= v
        if self.kappa:
            eps = {("x", "y"): 1, ("y", "z"): 1, ("z", "x"): 1,
                   ("y", "x"): -1, ("z", "y"): -1, ("x", "z"): -1}
            for (j, k, l, a, g, bb) in self.triples:
                v = -2.0 * self.kappa * eps[(a, bb)] * \
                    self.u_directed(j, k, u) * self.u_directed(k, l, u)
                A[j, l] += v
                A[l, j] -= v
        return A

    def spectrum(self, u: np.ndarray = None) -> np.ndarray:
        """Non-negative single-particle Majorana energies eps_m (N/2 values)."""
        A = self.A_matrix(u)
        if self.backend == "gpu":
            xp = xp_module("gpu")
            w = to_numpy(xp.linalg.eigvalsh(xp.asarray(1j * A)))
        else:
            w = np.linalg.eigvalsh(1j * A)
        return np.sort(w[w.size // 2:].clip(min=0.0))

    def matter_ground_energy(self, u: np.ndarray = None) -> float:
        return -0.5 * float(np.sum(self.spectrum(u)))

    # ---------------------------------------------------------------- #
    def matter_parity(self, u: np.ndarray = None) -> int:
        """Fermion parity P = prod_r (-i c_A c_B) of the matter ground state."""
        A = self.A_matrix(u)
        o = self.pair_order
        sgn, _ = pfaffian(A[np.ix_(o, o)])
        return 1 if np.real(sgn) > 0 else -1

    def is_physical(self, u: np.ndarray = None) -> bool:
        """Does the matter ground state in gauge u satisfy prod_j D_j = 1?"""
        u = self.u if u is None else np.asarray(u)
        val = self._const * int(np.prod(np.sign(u))) * self.matter_parity(u)
        return val == 1

    def physical_energy(self, u: np.ndarray = None, tol: float = 1e-10) -> float:
        """Lowest energy of a *physical* state in gauge sector u.

        If the matter vacuum has the wrong parity, one Bogoliubov fermion
        (the lowest eps) must be added.
        """
        eps = self.spectrum(u)
        E0 = -0.5 * float(eps.sum())
        if eps[0] < tol:          # zero mode: both parities degenerate
            return E0
        return E0 if self.is_physical(u) else E0 + float(eps[0])

    # ---------------------------------------------------------------- #
    #                       gauge / flux utilities                     #
    # ---------------------------------------------------------------- #
    def sector_gauge(self, s1: int = 1, s2: int = 1, u: np.ndarray = None) -> np.ndarray:
        """Gauge field with Wilson-loop sector (s1, s2): u -> u * s1^w1 s2^w2."""
        u = np.ones(self.nb) if u is None else np.array(u, float)
        for n, b in enumerate(self.bonds):
            u[n] *= (s1 ** abs(b.wind[0])) * (s2 ** abs(b.wind[1]))
        return u

    def fluxes(self, u: np.ndarray = None) -> np.ndarray:
        """W_p = prod_{<jk> in p, j in A} u_jk for every hexagon (+1 = no vison)."""
        u = self.u if u is None else u
        out = []
        for p in self.lat.plaquettes("hexagon"):
            s = p.sites
            w = 1.0
            for m in range(6):
                i, j = s[m], s[(m + 1) % 6]
                d = np.array(p.coords[(m + 1) % 6]) - np.array(p.coords[m])
                n = self._match_bond(i, j, d)
                w *= u[n]
            out.append(w)
        return np.array(out)

    def _match_bond(self, i, j, d):
        for n, b in enumerate(self.bonds):
            if (b.i, b.j) == (i, j) and np.allclose(b.d, d, atol=1e-6):
                return n
            if (b.i, b.j) == (j, i) and np.allclose(b.d, -d, atol=1e-6):
                return n
        raise KeyError((i, j))

    def ground_state_energy(self, physical: bool = True) -> float:
        """Minimum over the 4 flux-free topological sectors (Lieb: flux-free GS)."""
        es = []
        for s1 in (1, -1):
            for s2 in (1, -1):
                u = self.sector_gauge(s1, s2)
                es.append(self.physical_energy(u) if physical else self.matter_ground_energy(u))
        return min(es)

    def sector_energies(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        """Energies of the four flux-free sectors: vacuum, physical, parity."""
        out = {}
        for s1 in (1, -1):
            for s2 in (1, -1):
                u = self.sector_gauge(s1, s2)
                eps = self.spectrum(u)
                out[(s1, s2)] = {"E_vacuum": -0.5 * eps.sum(),
                                 "E_physical": self.physical_energy(u),
                                 "vacuum_is_physical": self.is_physical(u),
                                 "eps_min": eps[0]}
        return out

    def vison_pair_energy(self, bond_index: int = 0) -> float:
        """Energy cost of two adjacent fluxes (flip u on one bond), large clusters."""
        u0 = np.ones(self.nb)
        u1 = u0.copy()
        u1[bond_index] *= -1
        return self.matter_ground_energy(u1) - self.matter_ground_energy(u0)

    # ---------------------------------------------------------------- #
    def majorana_correlations(self, u: np.ndarray = None) -> np.ndarray:
        """<c_j c_k> in the matter ground state (= I + sign(iA))."""
        A = self.A_matrix(u)
        w, U = np.linalg.eigh(1j * A)
        return np.eye(self.N) + (U * np.sign(w)) @ U.conj().T

    def bond_correlations(self, u: np.ndarray = None) -> Dict[str, float]:
        """Average <sigma^g_j sigma^g_k> on x, y, z bonds (nonzero only for NN!)."""
        u = self.u if u is None else u
        C = self.majorana_correlations(u)
        acc = {"x": [], "y": [], "z": []}
        for n, b in enumerate(self.bonds):
            acc[b.kind].append(np.real(-1j * u[n] * C[b.i, b.j]))
        return {k: float(np.mean(v)) for k, v in acc.items()}

    # ---------------------------------------------------------------- #
    def bloch_hamiltonian(self, k: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        """2x2 Bloch matrix iA(k) for a translation-invariant gauge."""
        return kitaev_bloch(k, tuple(self.K[c] for c in "xyz"), self.kappa)


# --------------------------------------------------------------------------- #
#                 momentum space (uniform flux-free gauge)                    #
# --------------------------------------------------------------------------- #
_SQ3 = np.sqrt(3.0)
_DX = np.array([0.5, _SQ3 / 6])     # A -> B along x bond (qslab honeycomb)
_DY = np.array([-0.5, _SQ3 / 6])
_DZ = np.array([0.0, -1 / _SQ3])


def kitaev_bloch(k, K=(1.0, 1.0, 1.0), kappa: float = 0.0) -> np.ndarray:
    """Bloch Hamiltonian h(k) = i A(k) of the flux-free Kitaev model.

    Spectrum +-eps(k), eps = sqrt(|f|^2 + Delta^2) with
    f(k) = 2 sum_g K_g e^{i k.d_g},  Delta(k) = 4 kappa [sin k.a1 - sin k.a2 + sin k.(a2-a1)].
    """
    k = np.asarray(k, float)
    f = 2 * (K[0] * np.exp(1j * k @ _DX) + K[1] * np.exp(1j * k @ _DY)
             + K[2] * np.exp(1j * k @ _DZ))
    a1, a2 = np.array([1.0, 0.0]), np.array([0.5, _SQ3 / 2])
    D = 4 * kappa * (np.sin(k @ a1) - np.sin(k @ a2) + np.sin(k @ (a2 - a1)))
    return np.array([[D, 1j * f], [-1j * np.conj(f), -D]])


def kitaev_energy_tl(K=(1.0, 1.0, 1.0), kappa: float = 0.0, nk: int = 300) -> float:
    """Thermodynamic-limit ground-state energy per site (BZ integration)."""
    B = honeycomb(1, 1).reciprocal()
    t = (np.arange(nk) + 0.5) / nk
    tot = 0.0
    for t1 in t:
        for t2 in t:
            kk = t1 * B[0] + t2 * B[1]
            tot += np.linalg.eigvalsh(kitaev_bloch(kk, K, kappa))[1]
    # E = -1/2 sum_{eps>0} ; one positive band per cell (2 sites)
    return -0.5 * tot / nk ** 2 / 2


def chern_number_fhs(hk, B: np.ndarray, nk: int = 48, band: int = 0) -> float:
    """Fukui-Hatsugai-Suzuki lattice Chern number of ``band`` of h(k)."""
    def vec(t1, t2):
        _, v = np.linalg.eigh(hk(t1 * B[0] + t2 * B[1]))
        return v[:, band]
    V = [[vec(i / nk, j / nk) for j in range(nk)] for i in range(nk)]
    F = 0.0
    for i in range(nk):
        for j in range(nk):
            u1 = np.vdot(V[i][j], V[(i + 1) % nk][j])
            u2 = np.vdot(V[(i + 1) % nk][j], V[(i + 1) % nk][(j + 1) % nk])
            u3 = np.vdot(V[(i + 1) % nk][(j + 1) % nk], V[i][(j + 1) % nk])
            u4 = np.vdot(V[i][(j + 1) % nk], V[i][j])
            F += np.angle(u1 * u2 * u3 * u4)
    return F / (2 * np.pi)


def kitaev_phase(K: Sequence[float], kappa: float = 0.0) -> str:
    """Phase of the Kitaev model.

    'B (gapless)' if the triangle inequalities |K_a| <= |K_b| + |K_c| hold,
    otherwise 'A_x/A_y/A_z (gapped, toric code)'; with kappa != 0 the B phase
    becomes the 'B (chiral, Chern +-1)' non-Abelian phase.
    """
    a = np.abs(np.asarray(K, float))
    for n, lab in enumerate("xyz"):
        if a[n] > a.sum() - a[n] + 1e-14:
            return f"A_{lab} (gapped, Abelian toric-code)"
    return "B (chiral, non-Abelian Ising, Chern +-1)" if kappa else "B (gapless Majorana cones)"


# --------------------------------------------------------------------------- #
#                 finite temperature: Majorana Monte Carlo                    #
# --------------------------------------------------------------------------- #
def _free_energy_u(km: KitaevMajorana, u, beta):
    eps = km.spectrum(u)
    x = beta * eps / 2
    # -T sum ln(2 cosh x), numerically stable
    F = -np.sum(np.abs(x) + np.log1p(np.exp(-2 * np.abs(x)))) / beta
    E = -0.5 * np.sum(eps * np.tanh(x))
    varE = np.sum((eps / 2) ** 2 / np.cosh(np.minimum(x, 350)) ** 2)
    return F, E, varE


def _mc_one_temperature(args):
    (L1, L2, K, T, n_therm, n_meas, seed) = args
    rng = np.random.default_rng(seed)
    km = KitaevMajorana(honeycomb(L1, L2), K)
    beta = 1.0 / T
    # random start; random bond choice (a fixed permutation at infinite T would be a
    # global gauge flip that never changes W_p, an ergodicity trap)
    u = rng.choice([-1.0, 1.0], km.nb) if T > 0.5 else np.ones(km.nb)
    F, E, varE = _free_energy_u(km, u, beta)
    Es, E2s, Ws, acc = [], [], [], 0
    for sweep in range(n_therm + n_meas):
        for n in rng.integers(0, km.nb, km.nb):
            u[n] *= -1
            F1, E1, v1 = _free_energy_u(km, u, beta)
            if rng.random() < np.exp(-beta * (F1 - F)):
                F, E, varE = F1, E1, v1
                acc += 1
            else:
                u[n] *= -1
        if sweep >= n_therm:
            Es.append(E)
            E2s.append(E * E + varE)
            Ws.append(np.mean(km.fluxes(u)))
    Es, E2s = np.array(Es), np.array(E2s)
    N = km.N
    C = beta ** 2 * (E2s.mean() - Es.mean() ** 2) / N
    return {"T": T, "E": Es.mean() / N, "C": C, "W": float(np.mean(Ws)),
            "acc": acc / ((n_therm + n_meas) * km.nb)}


def majorana_monte_carlo(L: int = 4, temperatures=None, K=(1.0, 1.0, 1.0),
                         n_therm: int = 200, n_meas: int = 400, seed: int = 0,
                         n_jobs: Optional[int] = None, backend: str = "auto"):
    """Sign-free Monte Carlo over the Z2 gauge field (Nasu-Udagawa-Motome).

    Each temperature is an independent Markov chain, run in parallel on a
    process pool or over MPI ranks (``backend='auto'``).  Returns arrays T, E,
    C (per site), <W_p> and acceptance.  Entropy follows from integrating C/T.

    Expected physics (isotropic, Pauli units K = 1): a high-T peak of C near
    T_H ~ 1.5 K, where itinerant Majoranas freeze out and the entropy drops to
    (1/2) ln 2, and a low-T peak near T_L ~ 0.05 K, where the fluxes order
    (<W_p> -> 1).  In spin units with J = 4K these are T_H ~ 0.375 J and
    T_L ~ 0.012 J (Nasu et al.).  T_L is enhanced on small clusters (L <~ 6).
    """
    if temperatures is None:
        temperatures = np.geomspace(0.005, 5, 24)
    args = [(L, L, tuple(K), float(T), n_therm, n_meas, seed + 7919 * n)
            for n, T in enumerate(temperatures)]
    res = pmap(_mc_one_temperature, args, n_jobs=n_jobs, backend=backend)
    keys = res[0].keys()
    out = {k: np.array([r[k] for r in res]) for k in keys}
    o = np.argsort(out["T"])
    out = {k: v[o] for k, v in out.items()}
    # entropy by integrating C/T from infinite T (S = ln 2 per site)
    T = out["T"]
    S = np.zeros_like(T)
    S[-1] = np.log(2) - _tail_entropy(T[-1])
    lnT = np.log(T)
    for i in range(len(T) - 2, -1, -1):   # S = int C dlnT (trapezoid in ln T)
        S[i] = S[i + 1] - 0.5 * (out["C"][i] + out["C"][i + 1]) * (lnT[i + 1] - lnT[i])
    out["S"] = S
    return out


def _tail_entropy(T):
    # high-T expansion: C/N = (3/2) K^2 / T^2 (3/2 bonds per site, (K sigma sigma)^2 = K^2)
    # S(inf) - S(T) = int_T^inf C/T dT = (3/2)/(2 T^2)
    return 0.75 / T ** 2
