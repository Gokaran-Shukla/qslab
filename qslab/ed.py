"""
Exact diagonalization (ED) for spin-1/2 Hamiltonians.

Features
--------
* Full Hilbert space or fixed-magnetisation sectors (``n_up``), chosen
  automatically when ``H`` conserves S^z.
* Sparse-matrix (fast) or matrix-free (memory-light) Lanczos.
* Ground states, low-lying spectra, degeneracies and gaps.
* Quantum numbers of eigenstates: total spin S, crystal momentum k
  (from lattice translations), arbitrary symmetry operators (e.g. W_p).
* Observables: any :class:`~qslab.operators.Operator`, all spin
  correlations <S_i.S_j>, static structure factor S(q), dimer and chirality
  correlations.
* Entanglement entropy / entanglement spectrum for any bipartition.
* Finite temperature: exact (full spectrum) and finite-temperature Lanczos
  (FTLM, Jaklic-Prelovsek) - specific heat, entropy, susceptibility.
* Dynamics: dynamical correlation functions S(q, w) via continued-fraction
  Lanczos, real-time evolution.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

from .operators import (CompiledOperator, Operator, sdot, spin, total_spin)
from .parallel import HAVE_CUPY, HAVE_NUMBA, get_backend, get_num_threads, pmap, to_numpy

__all__ = ["SpinBasis", "ED", "lanczos_tridiag", "thermodynamics_from_spectrum",
           "ftlm_thermodynamics", "spectrum_all_sectors"]

DENSE_LIMIT = 2500


# --------------------------------------------------------------------------- #
class SpinBasis:
    """Computational basis of N spin-1/2, optionally fixed number of up spins.

    Bit i of the integer label = 0 for up, 1 for down (see operators.py).
    """

    def __init__(self, N: int, n_up: Optional[int] = None):
        if N > 62:
            raise ValueError("N > 62 not representable")
        self.N = N
        self.n_up = n_up
        if n_up is None:
            self.states = np.arange(2 ** N, dtype=np.uint64)
            self.full = True
        else:
            if not 0 <= n_up <= N:
                raise ValueError("n_up out of range")
            self.full = False
            self.states = self._sector_states(N, N - n_up)

    @staticmethod
    def _sector_states(N, n_dn):
        if N <= 26:
            allst = np.arange(2 ** N, dtype=np.uint64)
            return allst[np.bitwise_count(allst) == n_dn]
        # Gosper's hack (slow path for very large N)
        from itertools import combinations
        out = [sum(1 << i for i in c) for c in combinations(range(N), n_dn)]
        return np.array(sorted(out), dtype=np.uint64)

    @property
    def dim(self) -> int:
        return len(self.states)

    def __len__(self):
        return self.dim

    @property
    def sz(self) -> float:
        return None if self.n_up is None else self.n_up - self.N / 2

    def index(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Indices of integer states ``s`` and a mask telling which are in the basis."""
        s = np.asarray(s, dtype=np.uint64)
        if self.full:
            return s.astype(np.int64), np.ones(len(s), bool)
        idx = np.searchsorted(self.states, s)
        idx_c = np.minimum(idx, self.dim - 1)
        ok = self.states[idx_c] == s
        return idx_c, ok

    def lin_tables(self):
        """Cached O(1) index tables for the Numba kernel."""
        if not hasattr(self, "_lin"):
            from .kernels import lin_tables
            self._lin = lin_tables(self.states, self.N, self.N - self.n_up)
        return self._lin

    def device_states(self):
        """Basis integers on the GPU (cached)."""
        if not hasattr(self, "_dev"):
            import cupy as cp
            self._dev = cp.asarray(self.states)
        return self._dev

    def to_full(self, psi: np.ndarray) -> np.ndarray:
        """Embed a sector vector into the 2^N space."""
        if self.full:
            return psi
        out = np.zeros(2 ** self.N, dtype=psi.dtype)
        out[self.states.astype(np.int64)] = psi
        return out

    def spin_config(self, n: int) -> str:
        return "".join("d" if (n >> i) & 1 else "u" for i in range(self.N))

    def __repr__(self):
        sec = "full" if self.full else f"n_up={self.n_up} (Sz={self.sz:+g})"
        return f"<SpinBasis N={self.N} {sec} dim={self.dim}>"


def _xp_of(a):
    if HAVE_CUPY and not isinstance(a, np.ndarray):
        import cupy
        return cupy.get_array_module(a)
    return np


# --------------------------------------------------------------------------- #
def lanczos_tridiag(matvec, v0: np.ndarray, m: int, tol: float = 1e-12):
    """Plain Lanczos recursion; returns (alpha, beta) and norm of v0.

    No vectors are stored (O(dim) memory), which is all FTLM and
    continued-fraction dynamics need.
    """
    xp = _xp_of(v0)
    nrm = float(xp.linalg.norm(v0))
    v = v0 / nrm
    v_prev = xp.zeros_like(v)
    alpha, beta = [], []
    b = 0.0
    for _ in range(m):
        w = matvec(v)
        a = float(xp.vdot(v, w).real)
        w = w - a * v - b * v_prev
        alpha.append(a)
        b = float(xp.linalg.norm(w))
        if b < tol:
            break
        beta.append(b)
        v_prev, v = v, w / b
    return np.array(alpha), np.array(beta[:len(alpha) - 1]), nrm


# --------------------------------------------------------------------------- #
class ED:
    """Exact diagonalization driver.

    Parameters
    ----------
    H : Operator
        Hamiltonian (any Pauli-string operator).
    N : int
        Number of sites.
    n_up : int or None or 'auto'
        Sector with ``n_up`` up spins.  ``'auto'`` (default) picks
        the Sz = 0 (or 1/2) sector if H conserves S^z, else the full space.
    method : str
        How H is applied to vectors:

        ``'sparse'``  scipy CSR matrix (fast for dim up to about 1e7, but stores H)
        ``'numba'``   matrix-free parallel kernel on all cores (no H stored)
        ``'gpu'``     CuPy. Uses a cuSPARSE CSR matrix, or the matrix-free
                      kernel if ``matrix_free=True``
        ``'auto'``    gpu if backend is gpu, else sparse for dim < 2e6 and
                      numba above that
    matrix_free : bool
        Never build the sparse matrix (implies 'numba' on CPU).

    Example
    -------
    >>> from qslab import lattice, models, ED
    >>> lat = lattice.square(4)
    >>> ed = ED(models.heisenberg(lat, J1=1.0), lat.N)
    >>> E0, psi = ed.ground_state()
    >>> E0 / lat.N       # -0.70178 for the 4x4 square-lattice AFM
    """

    def __init__(self, H: Operator, N: int, n_up="auto", matrix_free: bool = False,
                 method: str = "auto"):
        self.H = H
        self.N = N
        if n_up == "auto":
            n_up = (N + 1) // 2 if H.conserves_sz() else None
            if n_up is not None and N % 2 == 1:
                n_up = (N + 1) // 2
        self.basis = SpinBasis(N, n_up)
        self.Hc = CompiledOperator(H, N)
        self.matrix_free = matrix_free
        if method == "auto":
            if get_backend() == "gpu":
                method = "gpu"
            elif matrix_free or (HAVE_NUMBA and self._csr_bytes_estimate() > 0.4 * _avail_mem()):
                method = "numba" if HAVE_NUMBA else "numpy"
            else:
                method = "sparse"
        if method == "sparse" and matrix_free:
            method = "numba" if HAVE_NUMBA else "numpy"
        if method == "gpu" and not HAVE_CUPY:
            raise RuntimeError("method='gpu' needs CuPy and a CUDA device")
        self.method = method
        if method in ("numba", "numpy"):
            self.matrix_free = True
        self._mat = None
        self._gpu_mat = None
        self.dtype = float if self.Hc.is_real else complex
        self.energies: Optional[np.ndarray] = None
        self.vectors: Optional[np.ndarray] = None

    def __repr__(self):
        return f"<ED N={self.N} {self.basis} method={self.method}>"

    # ---------------------------------------------------------------- #
    @property
    def dim(self):
        return self.basis.dim

    @property
    def matrix(self):
        if self._mat is None:
            self._mat = self.Hc.to_sparse(self.basis)
        return self._mat

    def matvec(self, v):
        """H @ v with the selected method (v on device for 'gpu')."""
        if self.method == "sparse":
            if (HAVE_NUMBA and self.dim > 20000 and get_num_threads() >= 4
                    and isinstance(v, np.ndarray) and v.ndim == 1):
                from .kernels import csr_matvec
                return csr_matvec(self.matrix, v)       # multithreaded
            return self.matrix @ v
        if self.method == "gpu":
            if not self.matrix_free:
                if self._gpu_mat is None:
                    import cupyx.scipy.sparse as csp
                    self._gpu_mat = csp.csr_matrix(self.matrix)
                return self._gpu_mat @ v
            return self.Hc.apply(self.basis, v, method="gpu")
        return self.Hc.apply(self.basis, v, method=self.method)

    def device_states(self):
        return self.basis.device_states()

    def _csr_bytes_estimate(self):
        nnz_row = len(self.Hc.groups)
        itemsize = 8 if self.Hc.is_real else 16
        return self.basis.dim * nnz_row * (itemsize + 4) * 2.5   # build overhead

    def _to_device(self, v):
        if self.method == "gpu":
            import cupy as cp
            return cp.asarray(v)
        return v

    def linear_operator(self):
        return spla.LinearOperator((self.dim, self.dim), matvec=self.matvec,
                                   dtype=self.dtype)

    # ---------------------------------------------------------------- #
    def diagonalize(self, k: int = 6, dense: Optional[bool] = None, tol: float = 1e-10,
                    v0=None):
        """Lowest ``k`` eigenpairs (sorted).  Stores them in ``.energies/.vectors``."""
        if dense is None:
            dense = self.dim <= DENSE_LIMIT
        if dense:
            Hd = self.matrix.toarray()
            w, v = sla.eigh(Hd)
            k = min(k, self.dim) if k else self.dim
            w, v = w[:k], v[:, :k]
        elif self.method == "gpu":
            import cupy as cp
            import cupyx.scipy.sparse.linalg as cspla
            k = min(k, self.dim - 2)
            op = cspla.LinearOperator((self.dim, self.dim), matvec=self.matvec,
                                      dtype=self.dtype)
            w, v = cspla.eigsh(op, k=k, which="SA", tol=tol,
                               ncv=min(self.dim - 1, max(4 * k + 8, 40)))
            w, v = cp.asnumpy(w), cp.asnumpy(v)
            o = np.argsort(w)
            w, v = w[o], v[:, o]
        else:
            k = min(k, self.dim - 2)
            op = self.matrix if self.method == "sparse" else self.linear_operator()
            if v0 is None:
                rng = np.random.default_rng(1234)
                v0 = rng.standard_normal(self.dim)
                if self.dtype == complex:
                    v0 = v0 + 1j * rng.standard_normal(self.dim)
            # generous Krylov space: resolves (topological) degeneracies reliably
            ncv = min(self.dim - 1, max(4 * k + 8, 40))
            w, v = spla.eigsh(op, k=k, which="SA", tol=tol, v0=v0, ncv=ncv)
            o = np.argsort(w)
            w, v = w[o], v[:, o]
        self.energies, self.vectors = w, v
        return w, v

    def ground_state(self, **kw) -> Tuple[float, np.ndarray]:
        w, v = self.diagonalize(k=kw.pop("k", 2), **kw)
        return w[0], v[:, 0]

    def multiplets(self, tol: float = 1e-6) -> List[Tuple[float, int]]:
        """Group stored energies into degenerate levels: [(E, degeneracy), ...]."""
        if self.energies is None:
            raise RuntimeError("call diagonalize() first")
        out = []
        for e in self.energies:
            if out and abs(e - out[-1][0]) < tol:
                out[-1] = (out[-1][0], out[-1][1] + 1)
            else:
                out.append((e, 1))
        return out

    # ---------------------------------------------------------------- #
    #                          observables                             #
    # ---------------------------------------------------------------- #
    def expect(self, op: Operator, psi: np.ndarray, phi: np.ndarray = None) -> complex:
        """<phi|op|psi>.  Operator must keep the vector inside the basis sector."""
        c = CompiledOperator(op, self.N)
        val = c.expectation(self.basis, psi, phi)
        return val.real if abs(val.imag) < 1e-10 else val

    def apply(self, op: Operator, psi: np.ndarray) -> np.ndarray:
        return CompiledOperator(op, self.N).apply(self.basis, psi)

    def energy(self, psi):
        return np.vdot(psi, self.matvec(psi)).real / np.vdot(psi, psi).real

    def total_spin(self, psi: np.ndarray) -> float:
        """Total spin S from <S_tot^2> = S(S+1)."""
        s2 = self.expect(total_spin(self.N), psi)
        return float(-0.5 + np.sqrt(0.25 + max(np.real(s2), 0.0)))

    def spin_correlations(self, psi: np.ndarray, component: str = "all") -> np.ndarray:
        """Matrix C_ij = <S_i . S_j>  (or <S^a_i S^a_j> for component a)."""
        N = self.N
        C = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                if i == j:
                    C[i, i] = 0.75 if component == "all" else 0.25
                    continue
                if component == "all":
                    op = sdot(i, j)
                else:
                    op = spin(i, component) * spin(j, component)
                C[i, j] = C[j, i] = np.real(self.expect(op, psi))
        return C

    def structure_factor(self, psi: np.ndarray, positions: np.ndarray,
                         qs: np.ndarray, component: str = "all",
                         C: np.ndarray = None) -> np.ndarray:
        """S(q) = (1/N) sum_ij e^{i q.(r_i - r_j)} <S_i . S_j>."""
        if C is None:
            C = self.spin_correlations(psi, component)
        qs = np.atleast_2d(qs)
        ph = np.exp(1j * qs @ positions.T)          # (nq, N)
        return np.real(np.einsum("qi,ij,qj->q", ph, C, ph.conj())) / self.N

    def dimer_correlations(self, psi, bonds, ref: int = 0) -> np.ndarray:
        """<D_ref D_b> - <D_ref><D_b> with D_b = S_i.S_j on bond b."""
        b0 = bonds[ref]
        D0 = sdot(b0.i, b0.j)
        d0 = np.real(self.expect(D0, psi))
        out = np.zeros(len(bonds))
        for n, b in enumerate(bonds):
            Db = sdot(b.i, b.j)
            out[n] = np.real(self.expect(D0 * Db, psi)) - d0 * np.real(self.expect(Db, psi))
        return out

    # ---------------------------------------------------------------- #
    #                       symmetry quantum numbers                   #
    # ---------------------------------------------------------------- #
    def permute(self, psi: np.ndarray, perm: np.ndarray) -> np.ndarray:
        """Apply the site permutation ``i -> perm[i]`` to a state."""
        s = self.basis.states
        new = np.zeros_like(s)
        for i, p in enumerate(perm):
            new |= ((s >> np.uint64(i)) & np.uint64(1)) << np.uint64(p)
        idx, ok = self.basis.index(new)
        out = np.zeros_like(psi)
        out[idx] = psi
        return out

    def momenta(self, lattice, vectors: np.ndarray = None, energies=None,
                tol: float = 1e-6):
        """Crystal momentum of each (multiplet of) eigenstate(s).

        Returns a list of ``(E, k_frac)`` where ``k_frac = (k.a1, k.a2)/2pi``
        (mod 1).  Degenerate multiplets are resolved by simultaneously
        diagonalising T1 and T2 inside the multiplet.
        """
        if vectors is None:
            vectors, energies = self.vectors, self.energies
        T1, T2 = lattice.translations()
        out = []
        i = 0
        n = vectors.shape[1]
        while i < n:
            j = i + 1
            while j < n and abs(energies[j] - energies[i]) < tol:
                j += 1
            V = vectors[:, i:j]
            M1 = V.conj().T @ np.column_stack([self.permute(V[:, a], T1) for a in range(j - i)])
            M2 = V.conj().T @ np.column_stack([self.permute(V[:, a], T2) for a in range(j - i)])
            # common eigenbasis of commuting unitaries
            _, U = np.linalg.eig(M1 + np.pi * M2 + 0.1234 * M1 @ M2)
            for c in range(j - i):
                u = U[:, c] / np.linalg.norm(U[:, c])
                l1 = u.conj() @ M1 @ u
                l2 = u.conj() @ M2 @ u
                k1 = (-np.angle(l1) / (2 * np.pi)) % 1.0
                k2 = (-np.angle(l2) / (2 * np.pi)) % 1.0
                out.append((energies[i], (round(k1, 6) % 1.0, round(k2, 6) % 1.0)))
            i = j
        return out

    def symmetry_eigenvalues(self, ops: Sequence[Operator], vectors=None, energies=None,
                             tol: float = 1e-6):
        """Eigenvalues of commuting symmetry operators within each multiplet."""
        if vectors is None:
            vectors, energies = self.vectors, self.energies
        out = []
        i, n = 0, vectors.shape[1]
        while i < n:
            j = i + 1
            while j < n and abs(energies[j] - energies[i]) < tol:
                j += 1
            V = vectors[:, i:j]
            mats = [V.conj().T @ np.column_stack([self.apply(o, V[:, a]) for a in range(j - i)])
                    for o in ops]
            G = sum((np.pi ** (m + 1) * 0.37 ** m) * M for m, M in enumerate(mats))
            _, U = np.linalg.eig(G)
            for c in range(j - i):
                u = U[:, c] / np.linalg.norm(U[:, c])
                out.append((energies[i], tuple(np.round(np.real(u.conj() @ M @ u), 6)
                                                for M in mats)))
            i = j
        return out

    # ---------------------------------------------------------------- #
    #                          entanglement                            #
    # ---------------------------------------------------------------- #
    def reduced_density_matrix(self, psi: np.ndarray, A: Sequence[int]) -> np.ndarray:
        A = list(A)
        B = [i for i in range(self.N) if i not in A]
        s = self.basis.states
        a = np.zeros(len(s), np.int64)
        b = np.zeros(len(s), np.int64)
        for k, site in enumerate(A):
            a |= (((s >> np.uint64(site)) & np.uint64(1)).astype(np.int64) << k)
        for k, site in enumerate(B):
            b |= (((s >> np.uint64(site)) & np.uint64(1)).astype(np.int64) << k)
        M = np.zeros((2 ** len(A), 2 ** len(B)), dtype=psi.dtype)
        M[a, b] = psi
        if M.shape[0] <= M.shape[1]:
            return M @ M.conj().T
        # trace out the smaller side the other way round (same nonzero spectrum)
        return (M.conj().T @ M).conj()

    def entanglement_spectrum(self, psi, A) -> np.ndarray:
        """Eigenvalues of rho_A, descending."""
        rho = self.reduced_density_matrix(psi / np.linalg.norm(psi), A)
        w = np.clip(np.linalg.eigvalsh(rho)[::-1], 0, None)
        return w

    def entanglement_entropy(self, psi, A, alpha: float = 1.0) -> float:
        """von Neumann (alpha = 1) or Renyi-alpha entropy of region A."""
        p = self.entanglement_spectrum(psi, A)
        p = p[p > 1e-15]
        if alpha == 1:
            return float(-np.sum(p * np.log(p)))
        return float(np.log(np.sum(p ** alpha)) / (1 - alpha))

    # ---------------------------------------------------------------- #
    #                        finite temperature                        #
    # ---------------------------------------------------------------- #
    def thermodynamics_exact(self, temperatures, observables: Dict[str, Operator] = None):
        """Full diagonalisation thermodynamics (dim <= ~ 2e4)."""
        Hd = self.matrix.toarray()
        w, v = sla.eigh(Hd)
        extra = {}
        if observables:
            for name, op in observables.items():
                O = CompiledOperator(op, self.N).to_sparse(self.basis)
                extra[name] = np.real(np.einsum("ij,ij->j", v.conj(), O @ v))
        return thermodynamics_from_spectrum(w, temperatures, self.N, extra)

    def ftlm(self, temperatures, R: int = 20, M: int = 120, seed: int = 0):
        """Finite-temperature Lanczos for this sector.

        Returns dict with sector weights needed to combine sectors:
        ``Z(T) e^{beta E_ref}``, ``<E>_num``, ``<E^2>_num`` and ``E_ref``.
        """
        T = np.asarray(temperatures, float)
        if self.method == "sparse":
            self.matrix  # build once before forking workers

        def one(r_seed):
            rng = np.random.default_rng(r_seed)
            r = rng.standard_normal(self.dim)
            if self.dtype == complex:
                r = r + 1j * rng.standard_normal(self.dim)
            r = self._to_device(r)
            a, b, _ = lanczos_tridiag(self.matvec, r, M)
            e, U = sla.eigh_tridiagonal(a, b) if len(b) else (a, np.ones((1, 1)))
            return e, np.abs(U[0, :]) ** 2

        seeds = [seed * 100003 + r for r in range(R)]
        # random vectors are independent: threads (kernels release the GIL)
        # or MPI ranks
        res = pmap(one, seeds, backend=("mpi" if _mpi_on() else "thread"),
                   n_jobs=(1 if self.method in ("numba", "gpu") else None))
        evs = [e for e, _ in res]
        wts = [w for _, w in res]
        Eref = min(e.min() for e in evs)
        Z = np.zeros_like(T)
        E1 = np.zeros_like(T)
        E2 = np.zeros_like(T)
        for e, w in zip(evs, wts):
            boltz = np.exp(-np.outer(1 / T, e - Eref)) * w     # (nT, m)
            Z += boltz.sum(1)
            E1 += (boltz * e).sum(1)
            E2 += (boltz * e ** 2).sum(1)
        fac = self.dim / R
        return {"Z": Z * fac, "E1": E1 * fac, "E2": E2 * fac, "Eref": Eref}

    # ---------------------------------------------------------------- #
    #                            dynamics                              #
    # ---------------------------------------------------------------- #
    def dynamical_correlation(self, op: Operator, psi0: np.ndarray, E0: float,
                              omegas, eta: float = 0.05, n_lanczos: int = 200):
        """S_O(w) = sum_n |<n|O|0>|^2 delta(w - (E_n - E0)), Lorentzian width eta.

        Continued-fraction Lanczos; O must map the sector onto itself
        (e.g. S^z_q).  Returns real array over ``omegas``.
        """
        phi = self._to_device(self.apply(op, psi0))
        nrm2 = float(_xp_of(phi).vdot(phi, phi).real)
        if nrm2 < 1e-30:
            return np.zeros(len(omegas))
        a, b, _ = lanczos_tridiag(self.matvec, phi, n_lanczos)
        e, U = sla.eigh_tridiagonal(a, b) if len(b) else (a, np.ones((1, 1)))
        w = np.abs(U[0]) ** 2 * nrm2
        om = np.asarray(omegas)[:, None]
        return np.sum(w * (eta / np.pi) / ((om - (e - E0)) ** 2 + eta ** 2), axis=1)

    def sq_omega(self, psi0, E0, positions, qs, omegas, component="z", eta=0.05,
                 n_lanczos=150):
        """Dynamical structure factor S^{aa}(q, w) for a list of q (rows)."""
        out = np.zeros((len(qs), len(omegas)))
        for n, q in enumerate(qs):
            Sq = Operator()
            for i, r in enumerate(positions):
                Sq = Sq + np.exp(1j * q @ r) * spin(i, component)
            Sq = Sq * (1 / np.sqrt(self.N))
            out[n] = self.dynamical_correlation(Sq, psi0, E0, omegas, eta, n_lanczos)
        return out

    def time_evolve(self, psi0: np.ndarray, times: np.ndarray) -> np.ndarray:
        """|psi(t)> = exp(-iHt)|psi0> on an equally spaced time grid."""
        times = np.asarray(times)
        A = -1j * self.matrix
        return spla.expm_multiply(A, psi0.astype(complex), start=times[0],
                                  stop=times[-1], num=len(times), endpoint=True)


def _avail_mem() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return float(line.split()[1]) * 1024
    except Exception:
        pass
    return 8e9


def _mpi_on():
    from .parallel import HAVE_MPI, mpi_info
    return HAVE_MPI and mpi_info()[1] > 1


# --------------------------------------------------------------------------- #
def thermodynamics_from_spectrum(w: np.ndarray, temperatures, N: int,
                                 observables: Dict[str, np.ndarray] = None,
                                 degeneracy: np.ndarray = None) -> Dict[str, np.ndarray]:
    """Thermodynamics per site from a full spectrum.

    Returns dict with T, E (per site), C (specific heat per site),
    S (entropy per site), F (free energy per site) and thermal averages of
    any diagonal observable passed in ``observables``.
    """
    T = np.asarray(temperatures, float)
    g = np.ones_like(w) if degeneracy is None else degeneracy
    E0 = w.min()
    bw = np.exp(-np.outer(1 / T, w - E0)) * g
    Z = bw.sum(1)
    E = (bw * w).sum(1) / Z
    E2 = (bw * w ** 2).sum(1) / Z
    C = (E2 - E ** 2) / T ** 2
    F = E0 - T * np.log(Z)
    S = (E - F) / T
    out = {"T": T, "E": E / N, "C": C / N, "S": S / N, "F": F / N}
    if observables:
        for k, o in observables.items():
            out[k] = (bw * o).sum(1) / Z
    return out


def ftlm_thermodynamics(H: Operator, N: int, temperatures, R: int = 20, M: int = 120,
                        seed: int = 0, use_sectors: Optional[bool] = None,
                        n_jobs: Optional[int] = None):
    """FTLM thermodynamics combining all S^z sectors (if conserved).

    Returns per-site E, C, S and (for S^z-conserving H) the uniform
    susceptibility chi = <(S^z_tot)^2>/(N T).
    """
    T = np.asarray(temperatures, float)
    if use_sectors is None:
        use_sectors = H.conserves_sz()
    sectors = list(range(N + 1)) if use_sectors else [None]
    flip_sym = all(sum(a in "yz" for _, a in k) % 2 == 0 for k in H.terms)
    if use_sectors and flip_sym:  # pi-rotation about x maps Sz -> -Sz
        uniq = [n for n in sectors if 2 * n >= N]
    else:
        uniq = sectors

    def job(n_up):
        ed = ED(H, N, n_up=n_up)
        if ed.dim <= 300:
            return {"Z": None, "w": np.linalg.eigvalsh(ed.matrix.toarray())}
        return ed.ftlm(T, R=R, M=M, seed=seed + (n_up or 0))

    res = dict(zip(uniq, pmap(job, uniq, backend=("mpi" if _mpi_on() else "thread"),
                              n_jobs=n_jobs)))
    parts = []
    for n_up in sectors:
        key = n_up if (n_up is None or n_up in res) else N - n_up
        parts.append((res[key], n_up))
    Eref = min(p["w"].min() if p.get("Z") is None else p["Eref"] for p, _ in parts)
    Z = np.zeros_like(T); E1 = np.zeros_like(T); E2 = np.zeros_like(T); M2 = np.zeros_like(T)
    for p, n_up in parts:
        if p.get("Z") is None:
            bw = np.exp(-np.outer(1 / T, p["w"] - Eref))
            z, e1, e2 = bw.sum(1), (bw * p["w"]).sum(1), (bw * p["w"] ** 2).sum(1)
        else:
            sh = np.exp(-(p["Eref"] - Eref) / T)
            z, e1, e2 = p["Z"] * sh, p["E1"] * sh, p["E2"] * sh
        Z += z; E1 += e1; E2 += e2
        if n_up is not None:
            M2 += z * (n_up - N / 2) ** 2
    E = E1 / Z
    C = (E2 / Z - E ** 2) / T ** 2
    F = Eref - T * np.log(Z)
    S = (E - F) / T
    out = {"T": T, "E": E / N, "C": C / N, "S": S / N}
    if use_sectors:
        out["chi"] = M2 / Z / (N * T)
    return out


def spectrum_all_sectors(H: Operator, N: int, k: int = 6) -> Dict[int, np.ndarray]:
    """Lowest k levels in every S^z >= 0 sector: {2Sz: energies}."""
    secs = list(range((N + 1) // 2, N + 1))

    def job(n_up):
        ed = ED(H, N, n_up=n_up)
        w, _ = ed.diagonalize(k=min(k, ed.dim))
        return w

    res = pmap(job, secs, backend=("mpi" if _mpi_on() else "thread"))
    return {2 * n - N: w for n, w in zip(secs, res)}
