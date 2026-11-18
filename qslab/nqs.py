"""
Neural-network quantum states: complex RBM + stochastic reconfiguration.

    psi(s) = exp(sum_i a_i s_i) prod_j 2 cosh(b_j + sum_i W_ji s_i),  s_i = +-1

(Carleo & Troyer, Science 355, 602 (2017)).  Works for any qslab Operator.

* ``sampler='exact'``      exact sums over the (sector) basis, no noise
  (N <~ 24).  Best for learning how sign structures and frustration
  affect variational wave functions.
* ``sampler='metropolis'`` Markov chains with spin-exchange (S^z fixed) or
  spin-flip moves.  Parallel over MPI ranks: each rank samples, and
  gradients and S-matrices are averaged with ``allreduce``.
* ``marshall=True`` adds the Marshall sign of bipartite lattices.  Without
  it the RBM has to learn the sign structure, and that becomes hard on
  frustrated (triangular/kagome) lattices.  This is a central lesson of
  the NQS tutorial.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .ed import SpinBasis
from .operators import CompiledOperator, Operator
from .parallel import allreduce_mean, mpi_info

__all__ = ["RBM", "NQS"]


def _spins(states: np.ndarray, N: int) -> np.ndarray:
    bits = ((states[:, None] >> np.arange(N, dtype=np.uint64)[None, :]) & np.uint64(1)).astype(np.int8)
    return 1.0 - 2.0 * bits


class RBM:
    def __init__(self, N: int, alpha: float = 2.0, seed: int = 0, scale: float = 0.01):
        rng = np.random.default_rng(seed)
        self.N, self.M = N, int(alpha * N)
        r = lambda *s: scale * (rng.standard_normal(s) + 1j * rng.standard_normal(s))
        self.a, self.b, self.W = r(N), r(self.M), r(self.M, N)

    @property
    def n_params(self):
        return self.N + self.M + self.M * self.N

    def get(self):
        return np.concatenate([self.a, self.b, self.W.ravel()])

    def set(self, p):
        N, M = self.N, self.M
        self.a, self.b, self.W = p[:N].copy(), p[N:N + M].copy(), p[N + M:].reshape(M, N).copy()

    def log_psi(self, s: np.ndarray) -> np.ndarray:
        theta = s @ self.W.T + self.b
        return s @ self.a + np.sum(_logcosh(theta), axis=1)

    def grad_log(self, s: np.ndarray) -> np.ndarray:
        theta = s @ self.W.T + self.b
        t = np.tanh(theta)
        return np.concatenate([s, t, (t[:, :, None] * s[:, None, :]).reshape(len(s), -1)], axis=1)


def _logcosh(x):
    # stable complex log cosh
    sgn = np.where(x.real >= 0, 1.0, -1.0)
    y = x * sgn
    return y + np.log1p(np.exp(-2 * y)) - np.log(2.0)


class NQS:
    """Variational ground-state search with an RBM.

    Parameters
    ----------
    H : Operator, N : int
    alpha : hidden-unit density M/N
    n_up : fix S^z sector (default: Sz = 0 if H conserves it)
    sampler : 'exact' | 'metropolis'
    marshall : None or array of +1/-1 per site; sites with -1 contribute the
               factor (-1)^{n_down} (use ``lattice.sublattice_sign()``)
    """

    def __init__(self, H: Operator, N: int, alpha: float = 2.0, n_up="auto",
                 sampler: str = "exact", marshall=None, seed: int = 0):
        self.H, self.N = H, N
        self.Hc = CompiledOperator(H, N)
        if n_up == "auto":
            n_up = N // 2 if H.conserves_sz() else None
        self.n_up = n_up
        self.sampler = sampler
        self.rbm = RBM(N, alpha, seed)
        self.rng = np.random.default_rng(seed + 17 * mpi_info()[0])
        self.msign = None
        if marshall is not None:
            self.msign = (np.asarray(marshall) < 0).astype(np.int64)
        if sampler == "exact":
            self.basis = SpinBasis(N, n_up)
            if self.basis.dim > 3_000_000:
                raise ValueError("Hilbert space too large for exact sampling")
        self.history = []

    # ---------------------------------------------------------------- #
    def log_psi_states(self, states):
        lp = self.rbm.log_psi(_spins(states, self.N))
        if self.msign is not None:
            nd = np.sum(((states[:, None] >> np.arange(self.N, dtype=np.uint64)) & np.uint64(1)).astype(np.int64)
                        * self.msign[None, :], axis=1)
            lp = lp + 1j * np.pi * (nd % 2)
        return lp

    def local_energy(self, states, logpsi, lookup=None):
        """E_loc(s) = sum_s' <s|H|s'> psi(s')/psi(s).  ``lookup`` = (basis, logpsi_all)
        reuses already-computed amplitudes (exact sampler)."""
        Eloc = np.zeros(len(states), complex)
        for fm in self.Hc.groups:
            amp = self.Hc.amplitudes(states, fm)
            nz = np.abs(amp) > 1e-14
            if not nz.any():
                continue
            new = states[nz] ^ np.uint64(fm)
            if lookup is not None:
                idx, ok = lookup[0].index(new)
                lp_new = np.where(ok, lookup[1][idx], -np.inf)
            else:
                lp_new = self.log_psi_states(new)
            Eloc[nz] += np.conj(amp[nz]) * np.exp(lp_new - logpsi[nz])
        return Eloc

    # ---------------------------------------------------------------- #
    def _samples_exact(self):
        st = self.basis.states
        lp = self.log_psi_states(st)
        w = np.exp(2 * (lp.real - lp.real.max()))
        w /= w.sum()
        return st, lp, w

    def _samples_mc(self, n_samples, n_chains=16, sweeps_between=1, n_therm=50):
        N = self.N
        if not hasattr(self, "_chains"):
            ch = []
            for _ in range(n_chains):
                if self.n_up is None:
                    bits = self.rng.integers(0, 2, N)
                else:
                    bits = np.array([0] * self.n_up + [1] * (N - self.n_up))
                    self.rng.shuffle(bits)
                ch.append(int(sum(int(b) << i for i, b in enumerate(bits))))
            self._chains = np.array(ch, dtype=np.uint64)
            self._mc_steps(n_therm * N)
        per = max(1, n_samples // len(self._chains))
        out = []
        for _ in range(per):
            self._mc_steps(sweeps_between * N)
            out.append(self._chains.copy())
        st = np.concatenate(out)
        lp = self.log_psi_states(st)
        return st, lp, np.full(len(st), 1.0 / len(st))

    def _mc_steps(self, n):
        N = self.N
        c = self._chains
        lp = self.log_psi_states(c)
        for _ in range(n):
            if self.n_up is None:
                flip = (np.uint64(1) << self.rng.integers(0, N, len(c)).astype(np.uint64))
            else:
                i = self.rng.integers(0, N, len(c)).astype(np.uint64)
                j = self.rng.integers(0, N, len(c)).astype(np.uint64)
                bi = (c >> i) & np.uint64(1)
                bj = (c >> j) & np.uint64(1)
                ok = bi != bj
                flip = np.where(ok, (np.uint64(1) << i) | (np.uint64(1) << j), np.uint64(0))
            prop = c ^ flip
            lpp = self.log_psi_states(prop)
            acc = self.rng.random(len(c)) < np.exp(2 * (lpp.real - lp.real))
            c = np.where(acc, prop, c)
            lp = np.where(acc, lpp, lp)
        self._chains = c

    # ---------------------------------------------------------------- #
    def step(self, lr: float = 0.05, diag_shift: float = 1e-3, n_samples: int = 2000):
        if self.sampler == "exact":
            st, lp, w = self._samples_exact()
        else:
            st, lp, w = self._samples_mc(n_samples)
        El = self.local_energy(st, lp, (self.basis, lp) if self.sampler == "exact" else None)
        O = self.rbm.grad_log(_spins(st, self.N))
        E = allreduce_mean(np.sum(w * El))
        Om = allreduce_mean(w @ O)
        Oc = O - Om
        F = allreduce_mean((w * (El - E)) @ Oc.conj())
        S = allreduce_mean((Oc.conj().T * w) @ Oc)
        S = S + diag_shift * np.diag(np.diag(S).real + 1e-8)
        dp = np.linalg.solve(S, F)
        self.rbm.set(self.rbm.get() - lr * dp)
        var = float(allreduce_mean(np.sum(w * np.abs(El - E) ** 2)))
        return E.real, var

    def optimize(self, n_iter: int = 300, lr: float = 0.05, diag_shift: float = 1e-3,
                 n_samples: int = 2000, verbose: int = 50, E_exact: Optional[float] = None):
        for it in range(n_iter):
            E, var = self.step(lr, diag_shift, n_samples)
            self.history.append((E, var))
            if verbose and (it % verbose == 0 or it == n_iter - 1) and mpi_info()[0] == 0:
                msg = f"iter {it:4d}  E={E:.6f}  E/N={E / self.N:.6f}  var={var:.2e}"
                if E_exact is not None:
                    msg += f"  rel.err={(E - E_exact) / abs(E_exact):.2e}"
                print(msg)
        return self.history[-1][0]

    def wavefunction(self):
        """Normalised vector in the exact basis (exact sampler only)."""
        lp = self.log_psi_states(self.basis.states)
        psi = np.exp(lp - lp.real.max())
        return psi / np.linalg.norm(psi)
