"""
Parton (Abrikosov-fermion) construction of spin liquids and Gutzwiller VMC.

    S_i = 1/2 f_i^dag sigma f_i,   n_i = 1

A U(1) spin-liquid ansatz is a spinon hopping problem
``H_MF = - sum_<ij> (chi_ij f^dag_i f_j + h.c.)`` with a pattern of gauge
fluxes through the plaquettes.  Projecting the half-filled spinon Fermi sea
onto singly occupied sites (Gutzwiller projection) gives a physical spin
wave function whose energy is evaluated

* **exactly**, by enumerating all configurations (N <~ 24): :func:`projected_state`;
* **stochastically**, by variational Monte Carlo with determinant ratios and
  Sherman-Morrison updates: :class:`GutzwillerVMC`.  Independent Markov
  chains run in parallel on processes or MPI ranks.

Flux patterns are imposed with a gauge-fixing solver
(:func:`hoppings_from_flux`).  It works on any lattice and any cluster
(real 0/pi fluxes via GF(2) elimination, arbitrary fluxes via least squares),
which gives

* uniform RVB / spinon Fermi surface (all fluxes 0),
* pi-flux state on the square lattice (Affleck-Marston),
* U(1) Dirac spin liquid on kagome [0, pi]: 0 through triangles, pi
  through hexagons (Ran, Hermele, Lee & Wen 2007; E ~ -0.4286 J/site),
* Dirac spin liquid on the triangular lattice (pi flux per unit cell),
* chiral spin liquids (flux phi through triangles), and so on.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .lattice import Lattice
from .parallel import allreduce_mean, mpi_info, pmap

__all__ = ["plaquette_incidence", "hoppings_from_flux", "SpinonAnsatz",
           "projected_state", "GutzwillerVMC", "config_sign"]


# --------------------------------------------------------------------------- #
#                              flux -> hoppings                               #
# --------------------------------------------------------------------------- #
def _bond_lookup(bonds):
    look = {}
    for n, b in enumerate(bonds):
        look.setdefault((b.i, b.j), []).append((n, np.array(b.d), +1))
        look.setdefault((b.j, b.i), []).append((n, -np.array(b.d), -1))
    return look


def plaquette_incidence(lat: Lattice, kinds: Sequence[str]) -> Tuple[np.ndarray, list]:
    """Oriented incidence matrix D[p, b] = +1/-1 if plaquette p traverses bond b
    along/against its (i -> j) direction (counter-clockwise loops)."""
    bonds = lat.bonds(1)
    look = _bond_lookup(bonds)
    plaqs = []
    for k in kinds:
        plaqs += lat.plaquettes(k)
    D = np.zeros((len(plaqs), len(bonds)), int)
    for r, p in enumerate(plaqs):
        s, c = p.sites, p.coords
        n = len(s)
        for m in range(n):
            i, j = s[m], s[(m + 1) % n]
            dv = np.array(c[(m + 1) % n]) - np.array(c[m])
            hit = [(idx, o) for (idx, d, o) in look.get((i, j), []) if np.allclose(d, dv, atol=1e-6)]
            if not hit:
                raise RuntimeError(f"edge {i}->{j} of plaquette not a NN bond")
            D[r, hit[0][0]] += hit[0][1]
    return D, plaqs


def _gf2_solve(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """One solution x of A x = b over GF(2) (None if inconsistent)."""
    A = (A % 2).astype(np.uint8).copy()
    b = (b % 2).astype(np.uint8).copy()
    m, n = A.shape
    piv_cols = []
    r = 0
    for c in range(n):
        rows = np.nonzero(A[r:, c])[0]
        if len(rows) == 0:
            continue
        p = r + rows[0]
        A[[r, p]] = A[[p, r]]
        b[[r, p]] = b[[p, r]]
        hit = np.nonzero(A[:, c])[0]
        hit = hit[hit != r]
        A[hit] ^= A[r]
        b[hit] ^= b[r]
        piv_cols.append(c)
        r += 1
        if r == m:
            break
    if np.any(b[r:]):
        return None
    x = np.zeros(n, np.uint8)
    for row, c in enumerate(piv_cols):
        x[c] = b[row]
    return x


def hoppings_from_flux(lat: Lattice, flux: Dict[str, float], twist=(0.0, 0.0),
                       magnitude: float = 1.0) -> np.ndarray:
    """Bond amplitudes chi_b (for lat.bonds(1)) realising the given fluxes.

    ``flux`` maps plaquette kind -> flux (radians) through each such plaquette,
    measured as arg prod chi around the counter-clockwise loop.  All kinds
    that tile the lattice must be given (e.g. kagome: up, down, hexagon).
    ``twist`` adds boundary phases (spinon boundary conditions / Wilson loops).
    """
    kinds = list(flux)
    D, plaqs = plaquette_incidence(lat, kinds)
    phi = np.array([flux[p.kind] for p in plaqs], float)
    tot = phi.sum() / (2 * np.pi)
    if all(lat.periodic) and abs(tot - round(tot)) > 1e-8:
        raise ValueError(f"total flux {tot:.4f} x 2pi is not an integer: "
                         "choose a cluster with an even number of unit cells")
    real = np.all(np.isclose(np.mod(phi, np.pi), 0) | np.isclose(np.mod(phi, np.pi), np.pi))
    if real:
        f = np.round(np.mod(phi, 2 * np.pi) / np.pi).astype(int) % 2
        x = _gf2_solve(np.abs(D), f)
        if x is None:
            raise ValueError("flux pattern inconsistent with this cluster")
        theta = np.pi * x
    else:
        # D theta = phi + 2 pi n ; fix integer offsets so the sum rule holds
        target = phi.copy()
        if all(lat.periodic):
            target[0] -= 2 * np.pi * round(tot)
        theta, *_ = np.linalg.lstsq(D.astype(float), target, rcond=None)
    bonds = lat.bonds(1)
    theta = theta + np.array([b.wind[0] * twist[0] + b.wind[1] * twist[1] for b in bonds])
    chi = magnitude * np.exp(1j * theta)
    if real and twist == (0.0, 0.0):
        chi = chi.real
    # check
    got = np.angle(np.exp(1j * (D @ np.angle(chi if np.iscomplexobj(chi) else chi.astype(complex)))))
    want = np.angle(np.exp(1j * phi))
    if not np.allclose(np.exp(1j * got), np.exp(1j * want), atol=1e-8):
        raise RuntimeError("flux solver failed")
    return chi


# --------------------------------------------------------------------------- #
class SpinonAnsatz:
    """Spinon mean-field ansatz H_MF = -sum_b (chi_b f^dag_i f_j + h.c.)

    Parameters
    ----------
    lat : Lattice
    chi : array of bond amplitudes for ``lat.bonds(1)`` (and optional
          further-neighbour dict ``extra = {order: amplitudes}``)
    """

    def __init__(self, lat: Lattice, chi, extra: Dict[int, np.ndarray] = None,
                 mu_stagger: np.ndarray = None):
        self.lat = lat
        self.N = lat.N
        self.chi = np.asarray(chi)
        h = np.zeros((self.N, self.N), complex)
        for b, c in zip(lat.bonds(1), self.chi):
            h[b.i, b.j] -= c
            h[b.j, b.i] -= np.conj(c)
        for order, amps in (extra or {}).items():
            for b, c in zip(lat.bonds(order), amps):
                h[b.i, b.j] -= c
                h[b.j, b.i] -= np.conj(c)
        if mu_stagger is not None:
            h += np.diag(mu_stagger)
        self.h = h if np.abs(h.imag).max() > 1e-12 else h.real
        self.eps, self.U = np.linalg.eigh(self.h)

    @classmethod
    def from_flux(cls, lat, flux, twist=(0.0, 0.0), **kw):
        return cls(lat, hoppings_from_flux(lat, flux, twist), **kw)

    # ---------------------------------------------------------------- #
    @property
    def n_occ(self):
        return self.N // 2

    def fermi_gap(self) -> float:
        """Gap between the highest occupied and lowest empty spinon level."""
        return float(self.eps[self.n_occ] - self.eps[self.n_occ - 1])

    def orbitals(self, check: bool = True) -> np.ndarray:
        """Occupied spinon orbitals (N x N/2).  Warns on open shells."""
        if check and self.fermi_gap() < 1e-8:
            import warnings
            warnings.warn("open shell at the Fermi level: the projected state is not "
                          "unique.  Use twisted boundary conditions (twist=...) "
                          "or a different cluster.")
        return self.U[:, :self.n_occ]

    def flux_through(self, kinds=None) -> Dict[str, np.ndarray]:
        kinds = kinds or list(self.lat._plaq_defs)
        D, plaqs = plaquette_incidence(self.lat, kinds)
        th = np.angle(self.chi.astype(complex))
        fl = np.angle(np.exp(1j * (D @ th)))
        out = {}
        for p, f in zip(plaqs, fl):
            out.setdefault(p.kind, []).append(f)
        return {k: np.array(v) for k, v in out.items()}

    def dos(self, omegas, eta=0.05):
        om = np.asarray(omegas)[:, None]
        return np.mean(eta / np.pi / ((om - self.eps) ** 2 + eta ** 2), axis=1)

    def mean_field_bond_energies(self) -> np.ndarray:
        """Unprojected <S_i.S_j> = -3/2 |G_ij|^2 (G_ij = <f^dag_i f_j>, per spin)."""
        G = self.orbitals(False) @ self.orbitals(False).conj().T
        return np.array([-1.5 * abs(G[b.i, b.j]) ** 2 for b in self.lat.bonds(1)])

    def dynamical_structure_factor(self, qs, omegas, eta=0.05):
        """Mean-field spinon continuum S^{zz}(q, w) (two-spinon particle-hole)."""
        occ = self.U[:, :self.n_occ]
        emp = self.U[:, self.n_occ:]
        eo, ee = self.eps[:self.n_occ], self.eps[self.n_occ:]
        r = self.lat.positions
        dE = (ee[None, :] - eo[:, None]).ravel()
        om = np.asarray(omegas)
        out = np.zeros((len(qs), len(om)))
        for n, q in enumerate(qs):
            ph = np.exp(1j * r @ q)
            M = occ.conj().T @ (ph[:, None] * emp)       # <m|e^{iqr}|n>
            w = (0.5 * np.abs(M) ** 2 * 2 / self.N).ravel()   # 2 spin species, S^z = sigma/2
            out[n] = np.sum(w[None, :] * (eta / np.pi) /
                            ((om[:, None] - dE[None, :]) ** 2 + eta ** 2), axis=1)
        return out


# --------------------------------------------------------------------------- #
#                 Gutzwiller projection: exact enumeration                    #
# --------------------------------------------------------------------------- #
def config_sign(states: np.ndarray, N: int) -> np.ndarray:
    """(-1)^{#(down at s, up at r, s < r)}: reorders site-ordered fermions
    c^dag_{1s1}...c^dag_{NsN} into (all ups)(all downs)."""
    par = np.zeros(len(states), np.int64)
    ndn = np.zeros(len(states), np.int64)
    for i in range(N):
        bit = ((states >> np.uint64(i)) & np.uint64(1)).astype(np.int64)  # 1 = down
        par += (1 - bit) * ndn
        ndn += bit
    return 1 - 2 * (par % 2)


def projected_state(ansatz: SpinonAnsatz, chunk: int = 20000):
    """Exact Gutzwiller-projected wave function in the S^z = 0 basis.

    Returns (basis, normalised vector).  Use with :class:`~qslab.ed.ED`
    to get exact variational energies and overlaps with the true ground state.
    """
    from .ed import SpinBasis
    N = ansatz.N
    if N % 2:
        raise ValueError("need even N")
    Phi = ansatz.orbitals()
    basis = SpinBasis(N, N // 2)
    st = basis.states
    psi = np.zeros(basis.dim, dtype=complex)
    sgn = config_sign(st, N)
    for a in range(0, basis.dim, chunk):
        s = st[a:a + chunk]
        bits = ((s[:, None] >> np.arange(N, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(bool)
        up_idx = np.array([np.nonzero(~row)[0] for row in bits])
        dn_idx = np.array([np.nonzero(row)[0] for row in bits])
        du = np.linalg.det(Phi[up_idx])
        dd = np.linalg.det(Phi[dn_idx])
        psi[a:a + chunk] = sgn[a:a + chunk] * du * dd
    nrm = np.linalg.norm(psi)
    if nrm < 1e-14:
        raise ValueError("projected state vanishes")
    psi /= nrm
    if np.abs(psi.imag).max() < 1e-12 * np.abs(psi).max():
        psi = psi.real.copy()
    return basis, psi


# --------------------------------------------------------------------------- #
#                         Variational Monte Carlo                             #
# --------------------------------------------------------------------------- #
class GutzwillerVMC:
    """Variational Monte Carlo for Gutzwiller-projected Fermi seas.

    H = sum_n J_n sum_<ij>_n S_i.S_j  (n = 1, 2, 3).

    Markov chain over S^z = 0 configurations with nearest-neighbour exchange
    moves.  Determinant ratios use the ``G = Phi M^-1`` trick, so all local
    energy ratios cost O(1) and an accepted move costs O(N^2).

    Parallelism: ``run(n_chains=...)`` distributes independent chains over
    a process pool or over MPI ranks (each rank runs its share and the
    results are merged).

    Example
    -------
    >>> lat = kagome(4, 4)
    >>> ans = SpinonAnsatz.from_flux(lat, {'up': 0, 'down': 0, 'hexagon': np.pi}, twist=(0, np.pi))
    >>> GutzwillerVMC(ans, J={1: 1.0}).run(n_sweeps=2000)['E_per_site']   # ~ -0.428
    """

    def __init__(self, ansatz: SpinonAnsatz, J: Dict[int, float] = None):
        self.ans = ansatz
        self.lat = ansatz.lat
        self.N = ansatz.N
        self.Phi = np.ascontiguousarray(ansatz.orbitals())
        self.J = J or {1: 1.0}
        self.pairs = []
        for order, Jv in self.J.items():
            for b in self.lat.bonds(order):
                self.pairs.append((b.i, b.j, Jv))
        self.pairs_arr = np.array([(i, j) for i, j, _ in self.pairs], int)
        self.Jarr = np.array([J for *_, J in self.pairs])
        self.move_pairs = [(b.i, b.j) for b in self.lat.bonds(1)]

    # ---------------------------------------------------------------- #
    def _init_config(self, rng):
        N = self.N
        for _ in range(5000):
            spins = np.array([0] * (N // 2) + [1] * (N // 2))   # 0 up, 1 down
            rng.shuffle(spins)
            up = list(np.nonzero(spins == 0)[0])
            dn = list(np.nonzero(spins == 1)[0])
            Mu, Md = self.Phi[up], self.Phi[dn]
            if np.linalg.cond(Mu) < 1e10 and np.linalg.cond(Md) < 1e10:
                return spins, up, dn, np.linalg.inv(Mu), np.linalg.inv(Md)
        raise RuntimeError("could not find a configuration with nonzero amplitude")

    def amplitude(self, spins) -> complex:
        """Exact (unnormalised) projected amplitude of a configuration."""
        up = np.nonzero(spins == 0)[0]
        dn = np.nonzero(spins == 1)[0]
        st = np.array([sum(int(b) << i for i, b in enumerate(spins))], dtype=np.uint64)
        return config_sign(st, self.N)[0] * np.linalg.det(self.Phi[up]) * np.linalg.det(self.Phi[dn])

    def _chain(self, seed: int, n_therm: int, n_sweeps: int, measure_every: int = 1):
        rng = np.random.default_rng(seed)
        N, Phi = self.N, self.Phi
        spins, up, dn, Mu_inv, Md_inv = self._init_config(rng)
        slot = np.zeros(N, int)       # slot of the particle on each site
        for k, s in enumerate(up):
            slot[s] = k
        for k, s in enumerate(dn):
            slot[s] = k
        Gu, Gd = Phi @ Mu_inv, Phi @ Md_inv          # (N, N/2)
        Es, acc, prop = [], 0, 0
        mp = self.move_pairs

        def ratio(i, j):
            # psi(i<->j)/psi for i up, j down.  Row replacement in slot order times
            # the reordering sign of the spin basis, which is always -1 for an
            # exchange (verified against exact amplitudes in tests)
            return -Gu[j, slot[i]] * Gd[i, slot[j]]

        def local_energy():
            e = 0.0
            for (i, j), Jv in zip(self.pairs_arr, self.Jarr):
                if spins[i] == spins[j]:
                    e += 0.25 * Jv
                else:
                    e -= 0.25 * Jv
                    a, b = (i, j) if spins[i] == 0 else (j, i)
                    e += 0.5 * Jv * ratio(a, b)
            return e

        n_moves = N
        for sweep in range(n_therm + n_sweeps):
            for _ in range(n_moves):
                i, j = mp[rng.integers(len(mp))]
                if spins[i] == spins[j]:
                    continue
                a, b = (i, j) if spins[i] == 0 else (j, i)   # a up, b down
                R = ratio(a, b)
                prop += 1
                if rng.random() < abs(R) ** 2:
                    acc += 1
                    ka, kb = slot[a], slot[b]
                    # Sherman-Morrison row replacement (up: row ka -> Phi[b])
                    Mu_inv = _sm_update(Mu_inv, Phi[b] - Phi[a], ka)
                    Md_inv = _sm_update(Md_inv, Phi[a] - Phi[b], kb)
                    spins[a], spins[b] = 1, 0
                    slot[b], slot[a] = ka, kb
                    Gu, Gd = Phi @ Mu_inv, Phi @ Md_inv
            if sweep >= n_therm and (sweep - n_therm) % measure_every == 0:
                Es.append(local_energy())
            if sweep % 50 == 0 and sweep:
                # refresh inverses to control round-off
                upl = np.argsort([slot[s] if spins[s] == 0 else 10 ** 9 for s in range(N)])[:N // 2]
                dnl = np.argsort([slot[s] if spins[s] == 1 else 10 ** 9 for s in range(N)])[:N // 2]
                Mu_inv = np.linalg.inv(Phi[upl])
                Md_inv = np.linalg.inv(Phi[dnl])
                Gu, Gd = Phi @ Mu_inv, Phi @ Md_inv
        Es = np.array(Es)
        return {"E": Es.real, "acc": acc / max(prop, 1)}

    # ---------------------------------------------------------------- #
    def run(self, n_sweeps: int = 1000, n_therm: int = 200, n_chains: int = None,
            seed: int = 0, backend: str = "auto", n_jobs: int = None) -> dict:
        """Energy of the projected state with a binning error bar.

        Under MPI each rank runs ``n_chains`` chains (default 1), so the total
        statistics scale with the number of ranks.
        """
        rank, size = mpi_info()
        if size > 1 and backend in ("auto", "mpi"):
            nc = n_chains or 1
            res = [self._chain(seed + 1009 * (rank * nc + c), n_therm, n_sweeps) for c in range(nc)]
            from .parallel import _COMM
            res = sum(_COMM.allgather(res), [])
        else:
            nc = n_chains or 1
            seeds = [seed + 1009 * c for c in range(nc)]
            res = pmap(lambda s: self._chain(s, n_therm, n_sweeps), seeds,
                       backend="thread" if backend == "auto" else backend, n_jobs=n_jobs)
        E = np.concatenate([r["E"] for r in res])
        chain_means = np.array([r["E"].mean() for r in res])
        err = _binning_error(E) if len(res) == 1 else chain_means.std(ddof=1) / np.sqrt(len(res))
        return {"E_per_site": E.mean() / self.N, "err_per_site": err / self.N,
                "E": E.mean(), "acceptance": np.mean([r["acc"] for r in res]),
                "n_samples": len(E), "chains": len(res)}


def _sm_update(Minv: np.ndarray, du: np.ndarray, k: int) -> np.ndarray:
    """Inverse after adding ``du`` to row k of M (Sherman-Morrison)."""
    # M' = M + e_k du^T ;  M'^-1 = M^-1 - (M^-1 e_k)(du^T M^-1)/(1 + du^T M^-1 e_k)
    col = Minv[:, k]
    row = du @ Minv
    return Minv - np.outer(col, row) / (1.0 + row[k])


def _binning_error(x: np.ndarray, min_bins: int = 16) -> float:
    x = np.asarray(x)
    errs = []
    b = 1
    while len(x) // b >= min_bins:
        nb = len(x) // b
        m = x[:nb * b].reshape(nb, b).mean(1)
        errs.append(m.std(ddof=1) / np.sqrt(nb))
        b *= 2
    return float(max(errs)) if errs else float(x.std() / np.sqrt(max(len(x), 1)))
