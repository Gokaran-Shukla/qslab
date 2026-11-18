"""
Pauli-string operator algebra and fast bit-level application.

Every Hamiltonian in qslab is an :class:`Operator`: a linear combination of
Pauli strings ``c * sigma^{a1}_{i1} sigma^{a2}_{i2} ...`` with at most one
Pauli matrix per site.  This single representation covers

* two-spin exchange of any form (Heisenberg, XXZ, Kitaev, Gamma, DM, ...),
* fields, scalar chirality S_i.(S_j x S_k), ring exchange,
* toric-code stars/plaquettes, Kitaev W_p fluxes, Wilson loops, ...

Basis convention (used everywhere in qslab)
-------------------------------------------
A basis state is an integer ``n``; bit ``i`` is 0 for spin up (sigma^z = +1)
and 1 for spin down.  With s(n_i) = (-1)^{n_i}::

    sigma^x |n> = |n ^ 2^i>
    sigma^y |n> = i (-1)^{n_i} |n ^ 2^i>
    sigma^z |n> =   (-1)^{n_i} |n>

so a Pauli string acts as ``P|n> = c * i^{n_y} * (-1)^{popcount(n & zmask)} |n ^ fmask>``
with ``fmask`` = sites carrying x or y and ``zmask`` = sites carrying y or z.
Grouping strings by ``fmask`` gives an extremely compact, vectorised
representation (:class:`CompiledOperator`) used by ED, NQS and VMC.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

__all__ = [
    "Operator", "CompiledOperator", "sigma", "spin", "identity",
    "splus", "sminus", "sdot", "exchange", "scalar_chirality",
    "pauli_string", "total_spin", "total_sz", "parity_sign",
]

# single-site Pauli multiplication table: (a, b) -> (phase, c)
_PMUL = {
    ("x", "x"): (1, ""), ("y", "y"): (1, ""), ("z", "z"): (1, ""),
    ("x", "y"): (1j, "z"), ("y", "z"): (1j, "x"), ("z", "x"): (1j, "y"),
    ("y", "x"): (-1j, "z"), ("z", "y"): (-1j, "x"), ("x", "z"): (-1j, "y"),
}
_TOL = 1e-13

Key = Tuple[Tuple[int, str], ...]


def _canon(ops: Iterable[Tuple[int, str]]) -> Tuple[complex, Key]:
    """Multiply a list of single-site Paulis into canonical (sorted) form."""
    per_site: Dict[int, str] = {}
    phase = 1.0 + 0j
    for site, a in ops:
        a = a.lower()
        if a not in "xyz" or len(a) != 1:
            raise ValueError(f"unknown Pauli {a!r}")
        site = int(site)
        if site in per_site and per_site[site]:
            ph, c = _PMUL[(per_site[site], a)]
            phase *= ph
            per_site[site] = c
        else:
            per_site[site] = a
    key = tuple(sorted((s, a) for s, a in per_site.items() if a))
    return phase, key


class Operator:
    """Linear combination of Pauli strings.

    Build with :func:`sigma`, :func:`spin` and arithmetic::

        H = sum(spin(i,'x')*spin(j,'x') for (i,j) in bonds)
        H = 0.5 * H + 2.0 * spin(0, 'z')

    ``Operator.terms`` maps a canonical key ``((site, 'x'), (site2, 'z'), ...)``
    to its complex coefficient.  The empty key is the identity.
    """

    __slots__ = ("terms",)

    def __init__(self, terms: Dict[Key, complex] = None):
        self.terms: Dict[Key, complex] = {}
        if terms:
            for k, v in terms.items():
                if abs(v) > _TOL:
                    self.terms[k] = complex(v)

    # ---------------------------------------------------------------- #
    @classmethod
    def from_string(cls, coeff, ops) -> "Operator":
        ph, key = _canon(ops)
        return cls({key: coeff * ph})

    def copy(self):
        return Operator(dict(self.terms))

    def __repr__(self):
        if not self.terms:
            return "Operator(0)"
        parts = []
        for k, v in list(self.terms.items())[:8]:
            s = " ".join(f"{a}{i}" for i, a in k) or "1"
            parts.append(f"({v.real:+.4g}{v.imag:+.4g}j) {s}")
        more = "" if len(self.terms) <= 8 else f" ... (+{len(self.terms) - 8} terms)"
        return "Operator[" + ", ".join(parts) + more + "]"

    def __len__(self):
        return len(self.terms)

    # ---------------------------------------------------------------- #
    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            other = identity() * other
        out = dict(self.terms)
        for k, v in other.terms.items():
            out[k] = out.get(k, 0) + v
        return Operator({k: v for k, v in out.items() if abs(v) > _TOL})

    __radd__ = __add__

    def __neg__(self):
        return Operator({k: -v for k, v in self.terms.items()})

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, (int, float, complex, np.number)):
            return Operator({k: v * other for k, v in self.terms.items()})
        out: Dict[Key, complex] = defaultdict(complex)
        for k1, v1 in self.terms.items():
            for k2, v2 in other.terms.items():
                ph, key = _canon(list(k1) + list(k2))
                out[key] += v1 * v2 * ph
        return Operator(dict(out))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, x):
        return self * (1.0 / x)

    def dagger(self) -> "Operator":
        # Pauli strings (canonical, one Pauli per site) are Hermitian
        return Operator({k: np.conj(v) for k, v in self.terms.items()})

    def is_hermitian(self, tol=1e-10) -> bool:
        return all(abs(v.imag) < tol for v in self.terms.values())

    def simplify(self, tol=1e-12) -> "Operator":
        return Operator({k: v for k, v in self.terms.items() if abs(v) > tol})

    def commutator(self, other) -> "Operator":
        return self * other - other * self

    def commutes_with(self, other, tol=1e-10) -> bool:
        c = self.commutator(other)
        return all(abs(v) < tol for v in c.terms.values())

    # ---------------------------------------------------------------- #
    @property
    def sites(self) -> List[int]:
        s = set()
        for k in self.terms:
            s.update(i for i, _ in k)
        return sorted(s)

    @property
    def max_locality(self) -> int:
        return max((len(k) for k in self.terms), default=0)

    def conserves_sz(self) -> bool:
        """True if [H, S^z_tot] = 0 (checked algebraically)."""
        sz = total_sz(max(self.sites, default=0) + 1)
        return self.commutes_with(sz)

    def compile(self, n_sites: int = None) -> "CompiledOperator":
        return CompiledOperator(self, n_sites)

    def to_matrix(self, n_sites: int):
        """Dense matrix in the full 2^N basis (small N only)."""
        from .ed import SpinBasis
        b = SpinBasis(n_sites)
        return self.compile(n_sites).to_sparse(b).toarray()


# ------------------------------------------------------------------------- #
#                            constructors                                   #
# ------------------------------------------------------------------------- #
def identity() -> Operator:
    return Operator({(): 1.0})


def sigma(i: int, a: str) -> Operator:
    """Pauli matrix sigma^a on site i."""
    return Operator.from_string(1.0, [(i, a)])


def spin(i: int, a: str) -> Operator:
    """Spin-1/2 operator S^a_i = sigma^a_i / 2."""
    return Operator.from_string(0.5, [(i, a)])


def splus(i: int) -> Operator:
    return spin(i, "x") + 1j * spin(i, "y")


def sminus(i: int) -> Operator:
    return spin(i, "x") - 1j * spin(i, "y")


def pauli_string(ops, coeff=1.0) -> Operator:
    """``pauli_string([(0,'x'),(3,'z')])`` = sigma^x_0 sigma^z_3."""
    return Operator.from_string(coeff, ops)


def sdot(i: int, j: int) -> Operator:
    """S_i . S_j"""
    return sum((spin(i, a) * spin(j, a) for a in "xyz"), Operator())


def exchange(i: int, j: int, J) -> Operator:
    """General bilinear exchange  sum_ab J[a,b] S^a_i S^b_j.

    ``J`` may be a scalar (Heisenberg), a length-3 vector (diagonal XYZ)
    or a full 3x3 matrix (Kitaev/Gamma/DM, ...).
    """
    J = np.asarray(J, dtype=complex)
    if J.ndim == 0:
        J = np.eye(3) * J
    elif J.ndim == 1:
        J = np.diag(J)
    out = Operator()
    for a, la in enumerate("xyz"):
        for b, lb in enumerate("xyz"):
            if abs(J[a, b]) > _TOL:
                out = out + J[a, b] * (spin(i, la) * spin(j, lb))
    return out


def scalar_chirality(i: int, j: int, k: int) -> Operator:
    """S_i . (S_j x S_k)"""
    eps = {("x", "y", "z"): 1, ("y", "z", "x"): 1, ("z", "x", "y"): 1,
           ("x", "z", "y"): -1, ("z", "y", "x"): -1, ("y", "x", "z"): -1}
    out = Operator()
    for (a, b, c), e in eps.items():
        out = out + e * (spin(i, a) * spin(j, b) * spin(k, c))
    return out


def total_sz(N: int) -> Operator:
    return sum((spin(i, "z") for i in range(N)), Operator())


def total_spin(N: int) -> Operator:
    """S_tot^2 = sum_ij S_i.S_j  (eigenvalue S(S+1))."""
    out = Operator()
    for a in "xyz":
        Sa = sum((spin(i, a) for i in range(N)), Operator())
        out = out + Sa * Sa
    return out


def parity_sign(x: np.ndarray) -> np.ndarray:
    """(-1)^{popcount(x)} for an integer array."""
    return 1 - 2 * (np.bitwise_count(x) & 1).astype(np.int8)


# ------------------------------------------------------------------------- #
#                         compiled representation                           #
# ------------------------------------------------------------------------- #
class CompiledOperator:
    """Operator grouped by flip mask for fast vectorised action.

    ``groups[fmask] = (zmasks, coeffs)``: on basis state ``n`` the operator
    produces ``sum_t coeffs[t] * (-1)^{popcount(n & zmasks[t])}`` times
    ``|n ^ fmask>``.
    """

    def __init__(self, op: Operator, n_sites: int = None):
        self.n_sites = n_sites if n_sites is not None else (max(op.sites, default=-1) + 1)
        groups: Dict[int, Dict[int, complex]] = defaultdict(lambda: defaultdict(complex))
        for key, c in op.terms.items():
            fm, zm, ny = 0, 0, 0
            for site, a in key:
                if site >= self.n_sites:
                    raise ValueError(f"operator acts on site {site} >= n_sites={self.n_sites}")
                bit = 1 << site
                if a in "xy":
                    fm |= bit
                if a in "yz":
                    zm |= bit
                if a == "y":
                    ny += 1
            groups[fm][zm] += c * (1j ** ny)
        self.groups = {}
        real = True
        for fm, d in groups.items():
            zms = np.array(list(d.keys()), dtype=np.uint64)
            cs = np.array(list(d.values()), dtype=complex)
            keep = np.abs(cs) > _TOL
            if keep.any():
                self.groups[fm] = (zms[keep], cs[keep])
                if np.any(np.abs(cs[keep].imag) > _TOL):
                    real = False
        self.is_real = real

    def amplitudes(self, states: np.ndarray, fmask: int) -> np.ndarray:
        zms, cs = self.groups[fmask]
        dtype = float if self.is_real else complex
        amp = np.zeros(len(states), dtype=dtype)
        for zm, c in zip(zms, cs):
            sgn = parity_sign(states & zm) if zm else 1
            amp += (c.real if self.is_real else c) * sgn
        return amp

    # ---------------------------------------------------------------- #
    def to_sparse(self, basis, check_closed: bool = True):
        """Sparse matrix ``<m|O|n>`` in a :class:`~qslab.ed.SpinBasis`."""
        import scipy.sparse as sp
        states = basis.states
        dim = len(states)
        rows, cols, vals = [], [], []
        col_all = np.arange(dim)
        for fm in self.groups:
            amp = self.amplitudes(states, fm)
            nz = np.abs(amp) > 1e-14
            if not nz.any():
                continue
            new = states[nz] ^ np.uint64(fm)
            idx, ok = basis.index(new)
            if check_closed and not ok.all():
                raise ValueError("operator does not conserve the chosen symmetry "
                                 "sector (e.g. S^z); use a larger basis")
            rows.append(idx[ok])
            cols.append(col_all[nz][ok])
            vals.append(amp[nz][ok])
        dtype = float if self.is_real else complex
        if not rows:
            return sp.csr_matrix((dim, dim), dtype=dtype)
        M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(dim, dim), dtype=dtype)
        M.sum_duplicates()
        return M

    # ---------------------------------------------------------------- #
    @property
    def flat(self):
        if not hasattr(self, "_flat"):
            from .kernels import flatten_groups
            self._flat = flatten_groups(self.groups, self.is_real)
        return self._flat

    def apply(self, basis, v, method: str = "auto"):
        """Matrix-free ``O @ v``.

        method: 'numba' (parallel CPU kernel), 'numpy', 'gpu' (CuPy) or
        'auto' (gpu if ``v`` is a CuPy array, numba if installed, else numpy).
        ``v`` may be 1D or 2D (columns are applied independently).
        """
        from . import kernels
        from .parallel import HAVE_NUMBA
        if method == "auto":
            if hasattr(v, "device") and not isinstance(v, np.ndarray):
                method = "gpu"
            elif HAVE_NUMBA and basis.dim >= 4096:
                method = "numba"
            else:
                method = "numpy"
        if v.ndim == 2:
            cols = [self.apply(basis, v[:, c], method) for c in range(v.shape[1])]
            if method == "gpu":
                import cupy
                return cupy.stack(cols, axis=1)
            return np.stack(cols, axis=1)
        fms, zoff, zms, cs = self.flat
        if method == "numba":
            tables = None if basis.full else basis.lin_tables()
            return kernels.apply_numba(basis.states, basis.full, fms, zoff, zms, cs, v, tables)
        if method == "gpu":
            import cupy as xp
            st = basis.device_states() if hasattr(basis, "device_states") else xp.asarray(basis.states)
            return kernels.apply_xp(xp, st, basis.full, fms, zoff, xp.asarray(zms),
                                    xp.asarray(cs), v)
        return kernels.apply_xp(np, basis.states, basis.full, fms, zoff, zms, cs, v)

    def expectation(self, basis, psi: np.ndarray, phi: np.ndarray = None) -> complex:
        """<phi|O|psi>  (phi defaults to psi), matrix-free."""
        if phi is None:
            phi = psi
        states = basis.states
        tot = 0.0 + 0j
        for fm in self.groups:
            amp = self.amplitudes(states, fm)
            nz = np.abs(amp) > 1e-14
            if not nz.any():
                continue
            new = states[nz] ^ np.uint64(fm)
            idx, ok = basis.index(new)
            src = np.nonzero(nz)[0][ok]
            tot += np.vdot(phi[idx[ok]], amp[nz][ok] * psi[src])
        return tot

    def connections(self, n: int):
        """All ``(n', <n'|O|n>)`` for a single basis integer (used by VMC/NQS)."""
        arr = np.array([n], dtype=np.uint64)
        out = []
        for fm in self.groups:
            a = self.amplitudes(arr, fm)[0]
            if abs(a) > 1e-14:
                out.append((n ^ fm, a))
        return out
