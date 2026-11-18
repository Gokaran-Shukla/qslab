"""
Two-site finite DMRG for arbitrary spin-1/2 Hamiltonians (CPU or GPU).

* The MPO is generated automatically from any :class:`~qslab.operators.Operator`
  with one- and two-site terms, including long-range bonds.  This covers 2D
  cylinders, where neighbours in the 1D site ordering can be far apart.
  Channels are shared by (start site, operator), so the MPO bond dimension is
  2 + (number of open channels).
* Lanczos effective-Hamiltonian solver, SVD truncation with ``chi_max`` and
  ``svd_cutoff``, bond-dimension schedules.
* Measurements: energy and variance, entanglement entropy on every bond,
  entanglement spectrum, local expectation values, two-point correlation
  matrices, arbitrary Operators through their MPO.
* Backends: NumPy with multithreaded BLAS on all cores of a node
  (``qslab.set_num_threads``) or CuPy on a GPU (``backend='gpu'``).

For production 2D work with U(1)/SU(2) symmetries use TeNPy, ITensor or block2.
This implementation is for learning, cross-checks and moderate bond
dimensions (chi up to about 1000 on a GPU).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .operators import Operator
from .parallel import get_backend, to_numpy, xp_module

__all__ = ["build_mpo", "DMRG", "lanczos_ground"]

_YR = np.array([[0.0, 1.0], [-1.0, 0.0]])   # sigma^y = -i * _YR (real)
_P = {
    "I": np.eye(2),
    "x": np.array([[0, 1], [1, 0]], complex),
    "y": np.array([[0, -1j], [1j, 0]], complex),
    "z": np.array([[1, 0], [0, -1]], complex),
}


def build_mpo(H: Operator, N: int, dtype=None) -> List[np.ndarray]:
    """MPO tensors W[k] with shape (w_left, w_right, d_out, d_in).

    Local basis: index 0 = up (sigma^z=+1), 1 = down (qslab convention).
    """
    # work with real matrices X, Yr = i sigma^y, Z: sigma^y = -i Yr, so a string
    # with n_y y's picks up (-i)^n_y.  The MPO is real whenever these
    # coefficients are (e.g. any Heisenberg/Kitaev/Gamma model).
    Q = {"x": _P["x"].real, "y": _YR, "z": _P["z"].real}
    singles: Dict[int, np.ndarray] = {}
    pairs = []   # (i, a, j, b, c), i < j
    const = 0.0
    coeffs = []
    for key, c in H.terms.items():
        ny = sum(a == "y" for _, a in key)
        c = c * (-1j) ** ny
        coeffs.append(c)
        if len(key) == 0:
            const += c
        elif len(key) == 1:
            (i, a), = key
            singles[i] = singles.get(i, 0) + c * Q[a]
        elif len(key) == 2:
            (i, a), (j, b) = key
            pairs.append((i, a, j, b, c))
        else:
            raise NotImplementedError("DMRG MPO builder supports 1- and 2-site terms; "
                                      "use ED for 3-spin (chirality/kappa) terms")
    is_complex = any(abs(np.imag(c)) > 1e-14 for c in coeffs)
    dt = dtype or (complex if is_complex else float)
    _Pl = Q

    last = {}
    for (i, a, j, b, c) in pairs:
        last[(i, a)] = max(last.get((i, a), -1), j)
    chans = []   # channels per bond k (between k and k+1)
    for k in range(N - 1):
        ch = sorted([key for key, jm in last.items() if key[0] <= k < jm])
        chans.append({key: 2 + n for n, key in enumerate(ch)})

    W = []
    for k in range(N):
        left = chans[k - 1] if k > 0 else {}
        right = chans[k] if k < N - 1 else {}
        wl, wr = 2 + len(left), 2 + len(right)
        Wk = np.zeros((wl, wr, 2, 2), dtype=complex)
        Wk[0, 0] = np.eye(2)
        Wk[1, 1] = np.eye(2)
        if k in singles:
            Wk[0, 1] += singles[k]
        if k == 0 and const:
            Wk[0, 1] += const * np.eye(2)
        for key, idx in right.items():
            i, a = key
            if i == k:
                Wk[0, idx] = _Pl[a]
            else:
                Wk[left[key], idx] = np.eye(2)
        for (i, a, j, b, c) in pairs:
            if j == k:
                Wk[left[(i, a)], 1] += c * _Pl[b]
        if dt is float:
            Wk = Wk.real
        W.append(Wk)
    W[0] = W[0][0:1]
    W[-1] = W[-1][:, 1:2]
    return W


def lanczos_ground(matvec, v0, xp=np, m: int = 30, tol: float = 1e-12, restarts: int = 4):
    """Lowest eigenpair by restarted Lanczos with full reorthogonalisation."""
    v = v0 / xp.linalg.norm(v0)
    E_old = None
    for _ in range(restarts):
        V = [v]
        alpha, beta = [], []
        w = matvec(v)
        for it in range(m):
            a = xp.vdot(V[-1], w).real
            alpha.append(float(a))
            w = w - a * V[-1] - (beta[-1] * V[-2] if beta else 0)
            # full reorthogonalisation
            for q in V:
                w = w - xp.vdot(q, w) * q
            b = float(xp.linalg.norm(w))
            if b < 1e-13 or it == m - 1:
                break
            beta.append(b)
            V.append(w / b)
            w = matvec(V[-1])
        T = np.diag(alpha) + np.diag(beta[:len(alpha) - 1], 1) + np.diag(beta[:len(alpha) - 1], -1)
        e, U = np.linalg.eigh(T)
        E = e[0]
        coef = U[:, 0]
        v = sum(float(c) * q for c, q in zip(coef, V))
        v = v / xp.linalg.norm(v)
        if E_old is not None and abs(E - E_old) < tol * max(1, abs(E)):
            break
        E_old = E
    return float(E), v


class DMRG:
    """Finite two-site DMRG.

    Parameters
    ----------
    H : Operator   (1- and 2-site Pauli terms; build with :mod:`qslab.models`)
    N : int        number of sites (MPS order = qslab site order)
    chi_max : int  maximal bond dimension (or use ``run(chi_schedule=...)``)
    svd_cutoff : discard Schmidt values below this
    backend : 'cpu' | 'gpu'
    init : 'random' | 'neel' | list of 0/1 local states

    Example
    -------
    >>> lat = qslab.lattice.triangular(8, 4, bc='cylinder')
    >>> d = DMRG(qslab.models.j1j2_triangular(lat, J2=0.1), lat.N, chi_max=400)
    >>> d.run(n_sweeps=10); d.energy / lat.N
    """

    def __init__(self, H: Operator, N: int, chi_max: int = 128, svd_cutoff: float = 1e-10,
                 backend: Optional[str] = None, init="random", seed: int = 0,
                 lanczos_m: int = 25):
        self.N = N
        self.H = H
        self.chi_max = chi_max
        self.cutoff = svd_cutoff
        self.backend = backend or get_backend()
        self.xp = xp_module(self.backend)
        Wn = build_mpo(H, N)
        self.dtype = Wn[0].dtype
        self.W = [self.xp.asarray(w) for w in Wn]
        self.lanczos_m = lanczos_m
        self.rng = np.random.default_rng(seed)
        self._init_mps(init)
        self.energy = None
        self.history: List[dict] = []
        self.entropies = np.zeros(N - 1)
        self.schmidt: List[np.ndarray] = [None] * (N - 1)

    # ---------------------------------------------------------------- #
    def _init_mps(self, init):
        xp = self.xp
        N = self.N
        if isinstance(init, str) and init == "random":
            chi = 4
            dims = [1] + [min(chi, 2 ** min(k, N - k)) for k in range(1, N)] + [1]
            M = []
            for k in range(N):
                t = self.rng.standard_normal((dims[k], 2, dims[k + 1]))
                if self.dtype == complex:
                    t = t + 1j * self.rng.standard_normal(t.shape)
                M.append(t)
        else:
            conf = [k % 2 for k in range(N)] if init == "neel" else list(init)
            M = []
            for k in range(N):
                t = np.zeros((1, 2, 1))
                t[0, conf[k], 0] = 1
                M.append(t.astype(self.dtype))
        self.M = [xp.asarray(m, dtype=self.dtype) for m in M]
        # right-canonicalise
        for k in range(N - 1, 0, -1):
            a, d, b = self.M[k].shape
            Q, R = xp.linalg.qr(self.M[k].reshape(a, d * b).T)
            self.M[k] = Q.T.reshape(-1, d, b)
            self.M[k - 1] = xp.tensordot(self.M[k - 1], R.T, axes=([2], [0]))
        self.M[0] = self.M[0] / xp.linalg.norm(self.M[0])
        self.L = [None] * N
        self.R = [None] * N
        self.L[0] = xp.ones((1, 1, 1), dtype=self.dtype)
        self.R[N - 1] = xp.ones((1, 1, 1), dtype=self.dtype)
        for k in range(N - 1, 0, -1):
            self.R[k - 1] = self._update_R(self.R[k], self.M[k], self.W[k])

    # ---------------------------------------------------------------- #
    def _update_L(self, L, A, W):
        xp = self.xp
        t = xp.tensordot(L, A, axes=([0], [0]))                 # w, a', s, b
        t = xp.tensordot(t, W, axes=([0, 2], [0, 3]))           # a', b, w', s'
        t = xp.tensordot(t, A.conj(), axes=([0, 3], [0, 1]))    # b, w', b'
        return t

    def _update_R(self, R, B, W):
        xp = self.xp
        t = xp.tensordot(B, R, axes=([2], [0]))                 # a, s, w', b'
        t = xp.tensordot(t, W, axes=([1, 2], [3, 1]))           # a, b', w, s'
        t = xp.tensordot(t, B.conj(), axes=([1, 3], [2, 1]))    # a, w, a'
        return t

    def _heff(self, k):
        xp = self.xp
        L, R, W1, W2 = self.L[k], self.R[k + 1], self.W[k], self.W[k + 1]
        shape = (L.shape[0], 2, 2, R.shape[0])

        def mv(x):
            th = x.reshape(shape)
            t = xp.tensordot(L, th, axes=([0], [0]))            # w, a', s1, s2, b
            t = xp.tensordot(t, W1, axes=([0, 2], [0, 3]))      # a', s2, b, v, s1'
            t = xp.tensordot(t, W2, axes=([3, 1], [0, 3]))      # a', b, s1', u, s2'
            t = xp.tensordot(t, R, axes=([1, 3], [0, 1]))       # a', s1', s2', b'
            return t.reshape(-1)
        return mv, shape

    def _optimize(self, k, chi, direction):
        xp = self.xp
        theta = xp.tensordot(self.M[k], self.M[k + 1], axes=([2], [0]))
        mv, shape = self._heff(k)
        E, v = lanczos_ground(mv, theta.reshape(-1), xp, m=self.lanczos_m)
        th = v.reshape(shape[0] * 2, 2 * shape[3])
        U, S, Vh = xp.linalg.svd(th, full_matrices=False)
        Sn = to_numpy(S)
        keep = int(min(chi, max(1, np.sum(Sn > self.cutoff * Sn[0]))))
        trunc = float(np.sum(Sn[keep:] ** 2))
        U, S, Vh = U[:, :keep], S[:keep], Vh[:keep]
        S = S / xp.linalg.norm(S)
        Sn = to_numpy(S)
        p = Sn ** 2
        p = p[p > 1e-16]
        self.entropies[k] = float(-np.sum(p * np.log(p)))
        self.schmidt[k] = Sn
        if direction == "right":
            self.M[k] = U.reshape(shape[0], 2, keep)
            self.M[k + 1] = (S[:, None] * Vh).reshape(keep, 2, shape[3])
            self.L[k + 1] = self._update_L(self.L[k], self.M[k], self.W[k])
        else:
            self.M[k] = (U * S[None, :]).reshape(shape[0], 2, keep)
            self.M[k + 1] = Vh.reshape(keep, 2, shape[3])
            self.R[k] = self._update_R(self.R[k + 1], self.M[k + 1], self.W[k + 1])
        return E, trunc

    # ---------------------------------------------------------------- #
    def sweep(self, chi: int = None):
        chi = chi or self.chi_max
        tr = 0.0
        for k in range(self.N - 1):
            E, t = self._optimize(k, chi, "right")
            tr = max(tr, t)
        for k in range(self.N - 2, -1, -1):
            E, t = self._optimize(k, chi, "left")
            tr = max(tr, t)
        self.energy = E
        return E, tr

    def run(self, n_sweeps: int = 8, chi_schedule: Sequence[int] = None,
            tol: float = 1e-9, verbose: bool = True):
        """Sweep until the energy converges.  ``chi_schedule`` grows chi per sweep."""
        if chi_schedule is None:
            chi_schedule = [min(self.chi_max, c) for c in (16, 32, 64, 128, 256, 512)]
            chi_schedule = sorted(set(chi_schedule)) + [self.chi_max] * n_sweeps
            chi_schedule = chi_schedule[:max(n_sweeps, 1)]
        E_old = None
        for s, chi in enumerate(chi_schedule):
            E, tr = self.sweep(chi)
            mx = max(m.shape[2] for m in self.M)
            self.history.append({"sweep": s, "chi": chi, "E": E, "trunc": tr, "max_bond": mx,
                                 "S_center": float(self.entropies[self.N // 2 - 1])})
            if verbose:
                print(f"sweep {s:2d}  chi={chi:4d}  E={E:.10f}  E/N={E / self.N:.8f}  "
                      f"trunc={tr:.1e}  S_mid={self.entropies[self.N // 2 - 1]:.4f}")
            if E_old is not None and abs(E - E_old) < tol * abs(E) and chi == chi_schedule[-1]:
                break
            E_old = E
        return self.energy

    # ---------------------------------------------------------------- #
    #                           measurements                           #
    # ---------------------------------------------------------------- #
    def _left_canonical(self):
        """Bring MPS to left-canonical form (norm on the last site)."""
        xp = self.xp
        for k in range(self.N - 1):
            a, d, b = self.M[k].shape
            Q, R = xp.linalg.qr(self.M[k].reshape(a * d, b))
            self.M[k] = Q.reshape(a, d, -1)
            self.M[k + 1] = xp.tensordot(R, self.M[k + 1], axes=([1], [0]))
        self.M[-1] = self.M[-1] / xp.linalg.norm(self.M[-1])

    def expectation_mpo(self, op: Operator) -> complex:
        """<psi|op|psi> for any 1-/2-site Operator via its MPO."""
        xp = self.xp
        Wo = [xp.asarray(w) for w in build_mpo(op, self.N, dtype=complex)]
        L = xp.ones((1, 1, 1), dtype=complex)
        for k in range(self.N):
            L = self._update_L(L, self.M[k].astype(complex), Wo[k])
        return complex(to_numpy(L).ravel()[0])

    def energy_variance(self) -> float:
        """<H^2> - <H>^2 (small = accurate eigenstate). Costs one H^2 MPO."""
        H2 = self.H * self.H
        try:
            e2 = self.expectation_mpo(H2).real
        except NotImplementedError:
            # H^2 has 4-site terms: contract MPO twice instead
            e2 = self._expect_two_mpos()
        e = self.expectation_mpo(self.H).real
        return e2 - e ** 2

    def _expect_two_mpos(self):
        xp = self.xp
        W = [xp.asarray(w).astype(complex) for w in build_mpo(self.H, self.N, dtype=complex)]
        L = xp.ones((1, 1, 1, 1), dtype=complex)
        for k in range(self.N):
            A = self.M[k].astype(complex)
            t = xp.tensordot(L, A, axes=([0], [0]))                     # w1 w2 a' s b
            t = xp.tensordot(t, W[k], axes=([0, 3], [0, 3]))             # w2 a' b v1 s1
            t = xp.tensordot(t, W[k], axes=([0, 4], [0, 3]))             # a' b v1 v2 s2
            t = xp.tensordot(t, A.conj(), axes=([0, 4], [0, 1]))         # b v1 v2 b'
            L = t
        return float(to_numpy(L).ravel()[0].real)

    def local_expectations(self, a: str = "z") -> np.ndarray:
        """<sigma^a_i> for all sites."""
        xp = self.xp
        self._left_canonical()
        P = xp.asarray(_P[a])
        # right environments of identity
        Rs = self._identity_right_envs()
        out = np.zeros(self.N)
        for i in range(self.N):
            A = self.M[i]
            t = xp.tensordot(A, P, axes=([1], [1]))                  # a b s'
            t = xp.tensordot(t, A.conj(), axes=([0, 2], [0, 1]))     # b b'
            out[i] = float(to_numpy(xp.sum(t * Rs[i])).real)
        return out

    def _identity_right_envs(self):
        xp = self.xp
        Rs = [None] * self.N
        R = xp.ones((1, 1), dtype=self.M[-1].dtype)
        for k in range(self.N - 1, -1, -1):
            Rs[k] = R
            B = self.M[k]
            R = xp.tensordot(xp.tensordot(B, R, axes=([2], [0])), B.conj(), axes=([1, 2], [1, 2]))
        return Rs

    def correlation_matrix(self, a: str = "z", b: str = None) -> np.ndarray:
        """C_ij = <sigma^a_i sigma^b_j> for all i, j (O(N^2 chi^3))."""
        b = b or a
        xp = self.xp
        self._left_canonical()
        Pa, Pb = xp.asarray(_P[a]), xp.asarray(_P[b])
        Rs = self._identity_right_envs()
        N = self.N
        C = np.zeros((N, N), complex)
        for i in range(N):
            A = self.M[i]
            # diagonal
            PP = Pa @ Pb
            t = xp.tensordot(xp.tensordot(A, PP, axes=([1], [1])), A.conj(), axes=([0, 2], [0, 1]))
            C[i, i] = complex(to_numpy(xp.sum(t * Rs[i])))
            T = xp.tensordot(xp.tensordot(A, Pa, axes=([1], [1])), A.conj(), axes=([0, 2], [0, 1]))
            for j in range(i + 1, N):
                Aj = self.M[j]
                tj = xp.tensordot(T, Aj, axes=([0], [0]))                   # b' s c
                val = xp.tensordot(tj, Pb, axes=([1], [1]))                 # b' c s'
                val = xp.tensordot(val, Aj.conj(), axes=([0, 2], [0, 1]))   # c c'
                C[i, j] = complex(to_numpy(xp.sum(val * Rs[j])))
                C[j, i] = np.conj(C[i, j]) if a == b else np.nan
                T = xp.tensordot(tj, Aj.conj(), axes=([0, 1], [0, 1]))      # transfer
        if a != b:
            for i in range(N):
                for j in range(i):
                    C[i, j] = np.nan
        return C.real if np.abs(C[np.isfinite(C)].imag).max(initial=0) < 1e-10 else C

    def spin_correlations(self) -> np.ndarray:
        """<S_i . S_j> matrix."""
        C = sum(np.real(self.correlation_matrix(a)) for a in "xyz") / 4.0
        return C

    def entanglement_spectrum(self, bond: int = None) -> np.ndarray:
        """-log(lambda^2) of the Schmidt values at a bond (default: center)."""
        bond = self.N // 2 - 1 if bond is None else bond
        s = self.schmidt[bond]
        return -2 * np.log(s[s > 1e-12])

    def bond_dimensions(self) -> List[int]:
        return [m.shape[2] for m in self.M[:-1]]
