"""
Model library: every spin-liquid Hamiltonian discussed in the tutorials.

All builders return an :class:`~qslab.operators.Operator` that works with
every solver (ED, DMRG, NQS; VMC needs Heisenberg-type models).  Spin
models use spin-1/2 operators S = sigma/2 unless stated otherwise.

=====================  ======================================================
builder                Hamiltonian
=====================  ======================================================
heisenberg             J1 sum_<ij> S_i.S_j + J2 sum_<<ij>> + J3 ... (+ XXZ, field)
j1j2_square            square-lattice J1-J2 (plaquette VBS / spin liquid near J2 ~ 0.5)
j1j2_triangular        triangular J1-J2 (spin liquid window ~0.07 < J2 < 0.15)
kagome_heisenberg      kagome J1 (+J2, J3, chiral term)  - Z2 / Dirac spin liquid
kitaev                 Kitaev honeycomb  -sum K_g sigma^g sigma^g (+ kappa 3-spin term)
kitaev_heisenberg_gamma  J-K-Gamma-Gamma'-J3 model for alpha-RuCl3, Na2IrO3
toric_code             -Je sum A_s - Jm sum B_p (+ fields)  - exact Z2 topological order
xxz_chain              1D benchmark with Bethe-ansatz energy
=====================  ======================================================
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .lattice import Lattice, square
from .operators import (Operator, exchange, pauli_string, scalar_chirality,
                        sigma, spin)

__all__ = ["heisenberg", "j1j2_square", "j1j2_triangular", "kagome_heisenberg",
           "kitaev", "kitaev_heisenberg_gamma", "kitaev_plaquette_operators",
           "kitaev_wilson_loop", "toric_code", "ToricCodeLattice",
           "xxz_chain", "zeeman", "chirality_terms", "kitaev_kappa_triples",
           "bethe_energy_per_site", "add_twist"]


# --------------------------------------------------------------------------- #
#                              helper terms                                   #
# --------------------------------------------------------------------------- #
def zeeman(N: int, h: Sequence[float], g: float = 1.0) -> Operator:
    """- g h . sum_i S_i"""
    h = np.asarray(h, float)
    out = Operator()
    for i in range(N):
        for a, la in enumerate("xyz"):
            if h[a]:
                out = out - g * h[a] * spin(i, la)
    return out


def chirality_terms(lat: Lattice, kinds=("up", "down")) -> Operator:
    """sum over triangles of S_i.(S_j x S_k), vertices counter-clockwise."""
    out = Operator()
    for k in kinds:
        for p in lat.plaquettes(k):
            i, j, l = p.sites
            out = out + scalar_chirality(i, j, l)
    return out


def add_twist(lat: Lattice, bonds, phi=(0.0, 0.0)):
    """Twist angles (U(1) about z) for bonds wrapping the torus.

    Returns per-bond phase e^{i(w1 phi1 + w2 phi2)}; useful for
    twisted-boundary averaging of the XXZ part.
    """
    return [np.exp(1j * (b.wind[0] * phi[0] + b.wind[1] * phi[1])) for b in bonds]


# --------------------------------------------------------------------------- #
#                          Heisenberg-type models                             #
# --------------------------------------------------------------------------- #
def heisenberg(lat: Lattice, J1: float = 1.0, J2: float = 0.0, J3: float = 0.0,
               Delta: float = 1.0, h: Sequence[float] = (0, 0, 0),
               Jchi: float = 0.0, D: Sequence[float] = None) -> Operator:
    """Heisenberg / XXZ model with up to third neighbours.

    H = sum_n J_n sum_<ij>_n [S^x S^x + S^y S^y + Delta S^z S^z]
        + Jchi sum_triangles S_i.(S_j x S_k)
        + sum_<ij> D.(S_i x S_j)      (DM vector on NN bonds, direction i->j)
        - h . sum_i S_i
    """
    H = Operator()
    for order, J in ((1, J1), (2, J2), (3, J3)):
        if J == 0:
            continue
        for b in lat.bonds(order):
            H = H + exchange(b.i, b.j, [J, J, J * Delta])
    if D is not None:
        D = np.asarray(D, float)
        eps = np.zeros((3, 3, 3))
        eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
        eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1
        Jdm = np.einsum("c,cab->ab", D, eps)    # D.(Si x Sj) = sum_ab Jdm_ab S^a_i S^b_j
        for b in lat.bonds(1):
            H = H + exchange(b.i, b.j, Jdm)
    if Jchi:
        H = H + Jchi * chirality_terms(lat)
    if np.any(np.asarray(h) != 0):
        H = H + zeeman(lat.N, h)
    return H


def j1j2_square(lat: Lattice, J1: float = 1.0, J2: float = 0.5, **kw) -> Operator:
    """Square-lattice J1-J2 Heisenberg model.

    Neel order for J2 < ~0.45; a debated gapless spin liquid / plaquette VBS
    for 0.45 - 0.6; stripe order beyond.
    """
    assert lat.name == "square"
    return heisenberg(lat, J1, J2, **kw)


def j1j2_triangular(lat: Lattice, J1: float = 1.0, J2: float = 0.1, **kw) -> Operator:
    """Triangular J1-J2 Heisenberg model.

    120-degree order for J2 < ~0.07, spin liquid for ~0.07-0.15 (likely the
    U(1) Dirac spin liquid), stripe order beyond.
    """
    assert lat.name == "triangular"
    return heisenberg(lat, J1, J2, **kw)


def kagome_heisenberg(lat: Lattice, J1: float = 1.0, J2: float = 0.0, J3: float = 0.0,
                      Jchi: float = 0.0, **kw) -> Operator:
    """Kagome Heisenberg antiferromagnet.

    J1 only: the archetypal QSL candidate (DMRG: gapped Z2, Yan-Huse-White 2011;
    VMC / later DMRG: U(1) Dirac spin liquid).  J2, J3 or Jchi > 0
    stabilise a chiral spin liquid (He, Sheng, Chen 2014; Gong et al. 2014).
    J3 here is the third-neighbour coupling across hexagons.
    """
    assert lat.name == "kagome"
    return heisenberg(lat, J1, J2, J3, Jchi=Jchi, **kw)


def xxz_chain(L: int, Delta: float = 1.0, J: float = 1.0, bc: str = "periodic",
              h: float = 0.0) -> Tuple[Lattice, Operator]:
    from .lattice import chain
    lat = chain(L, bc)
    return lat, heisenberg(lat, J, Delta=Delta, h=(0, 0, h))


def bethe_energy_per_site(L: int = None) -> float:
    """Ground-state energy per site of the S=1/2 Heisenberg chain (L -> inf).

    E/N = 1/4 - ln 2 = -0.443147...
    """
    return 0.25 - np.log(2)


# --------------------------------------------------------------------------- #
#                            Kitaev physics                                   #
# --------------------------------------------------------------------------- #
def kitaev_kappa_triples(lat: Lattice):
    """Three-spin terms of Kitaev's magnetic-field perturbation.

    For each site k and each pair of its bonds (j-k of type alpha, k-l of
    type beta) one term sigma^alpha_j sigma^gamma_k sigma^beta_l with gamma
    the third label.  Returns list of (j, k, l, alpha, gamma, beta).
    """
    nbr = {i: [] for i in range(lat.N)}
    for b in lat.bonds(1):
        nbr[b.i].append((b.j, b.kind))
        nbr[b.j].append((b.i, b.kind))
    out = []
    for k in range(lat.N):
        nb = nbr[k]
        for x in range(len(nb)):
            for y in range(x + 1, len(nb)):
                (j, a), (l, bb) = nb[x], nb[y]
                if a == bb:
                    continue
                g = ({"x", "y", "z"} - {a, bb}).pop()
                out.append((j, k, l, a, g, bb))
    return out


def kitaev(lat: Lattice, K: Sequence[float] = (1.0, 1.0, 1.0), kappa: float = 0.0,
           h: Sequence[float] = (0, 0, 0), units: str = "pauli") -> Operator:
    """Kitaev honeycomb model.

    units='pauli' (Kitaev's convention, default)::

        H = - sum_<ij>_g K_g sigma^g_i sigma^g_j - kappa sum sigma^a_j sigma^g_k sigma^b_l
            - h . sum_i sigma_i

    units='spin'::  H = sum_<ij>_g K_g S^g_i S^g_j  (K > 0 antiferromagnetic,
    K < 0 ferromagnetic, as in alpha-RuCl3 literature), field -h.S

    Exactly solvable for h = 0 (any K, kappa) by Majorana fermions; see
    :mod:`qslab.kitaev`.  Isotropic ground-state energy:
    E/N = -0.78730 K (pauli units) = -0.19682 |K| (spin units).
    """
    assert lat.name == "honeycomb", "Kitaev model lives on the honeycomb lattice"
    Kd = dict(zip("xyz", K))
    H = Operator()
    for b in lat.bonds(1):
        if units == "pauli":
            H = H - Kd[b.kind] * (sigma(b.i, b.kind) * sigma(b.j, b.kind))
        else:
            H = H + Kd[b.kind] * (spin(b.i, b.kind) * spin(b.j, b.kind))
    if kappa:
        for (j, k, l, a, g, bb) in kitaev_kappa_triples(lat):
            H = H - kappa * pauli_string([(j, a), (k, g), (l, bb)])
    h = np.asarray(h, float)
    if np.any(h):
        for i in range(lat.N):
            for c, lc in enumerate("xyz"):
                if h[c]:
                    H = H - h[c] * (sigma(i, lc) if units == "pauli" else spin(i, lc))
    return H


def kitaev_plaquette_operators(lat: Lattice):
    """Conserved fluxes W_p = sigma^x_1 sigma^y_2 sigma^z_3 sigma^x_4 sigma^y_5 sigma^z_6.

    For each hexagon, the operator on a vertex is the label of the bond
    that points *out* of the hexagon.  [W_p, H_Kitaev] = 0, W_p^2 = 1, and
    the ground state is flux-free (W_p = +1, Lieb's theorem).
    """
    bonds = lat.bonds(1)
    pairs = {}
    for b in bonds:
        pairs.setdefault(frozenset((b.i, b.j)), []).append(b)
    ops = []
    for p in lat.plaquettes("hexagon"):
        s = list(p.sites)
        in_labels = {}
        for m in range(6):
            i, j = s[m], s[(m + 1) % 6]
            P = np.array(p.coords[m]); Q = np.array(p.coords[(m + 1) % 6])
            # bond between consecutive vertices with matching displacement
            for b in pairs.get(frozenset((i, j)), []):
                d = np.array(b.d) if b.i == i else -np.array(b.d)
                if np.allclose(d, Q - P, atol=1e-6):
                    in_labels.setdefault(i, set()).add(b.kind)
                    in_labels.setdefault(j, set()).add(b.kind)
        string = []
        for v in s:
            out = ({"x", "y", "z"} - in_labels[v]).pop()
            string.append((v, out))
        ops.append(pauli_string(string))
    return ops


def kitaev_wilson_loop(lat: Lattice, direction: int = 0) -> Operator:
    """Non-contractible Wilson loop of the Kitaev model on a torus.

    A straight zig-zag string of sites along T1 (direction 0) or T2: on each
    site the operator is the label of the bond *not* used by the loop.  It
    commutes with H and all W_p and distinguishes the 4 topological sectors.
    """
    assert lat.rectangular and all(lat.periodic)
    bonds = lat.bonds(1)
    # path: along a1 use x and z bonds alternately; along a2 use y and z.
    use = ("x", "z") if direction == 0 else ("y", "z")
    adj = {}
    for b in bonds:
        if b.kind in use:
            adj.setdefault(b.i, []).append((b.j, b.kind, np.array(b.d)))
            adj.setdefault(b.j, []).append((b.i, b.kind, -np.array(b.d)))
    start = 0
    path = [start]
    used = []
    cur, prev = start, None
    while True:
        nxt = None
        for (j, kind, d) in adj[cur]:
            if (kind not in [u for u in used[-1:]]) and j != prev:
                nxt = (j, kind)
                break
        if nxt is None:
            raise RuntimeError("could not build Wilson loop")
        used.append(nxt[1])
        prev, cur = cur, nxt[0]
        if cur == start:
            break
        path.append(cur)
        if len(path) > lat.N:
            raise RuntimeError("Wilson loop did not close")
    # label on each site = bond type not in the two loop bonds through it
    string = []
    for m, v in enumerate(path):
        b_in = used[m - 1]
        b_out = used[m]
        lab = ({"x", "y", "z"} - {b_in, b_out}).pop()
        string.append((v, lab))
    return pauli_string(string)


def kitaev_heisenberg_gamma(lat: Lattice, J: float = 0.0, K: float = -1.0,
                            Gamma: float = 0.0, Gamma_p: float = 0.0,
                            J3: float = 0.0, h: Sequence[float] = (0, 0, 0),
                            g: float = 1.0) -> Operator:
    """Generalised Kitaev model (Rau, Lee & Kee 2014), spin-1/2 units.

    On a gamma-bond (alpha, beta the other two labels)::

        J S_i.S_j + K S^g_i S^g_j + Gamma (S^a_i S^b_j + S^b_i S^a_j)
                  + Gamma' (S^g_i S^a_j + S^a_i S^g_j + S^g_i S^b_j + S^b_i S^g_j)

    plus a third-neighbour Heisenberg J3 and Zeeman term -g h.S in the
    cubic (x,y,z) frame.  For alpha-RuCl3 a common parameter set is
    (J, K, Gamma, Gamma', J3) = (-0.1, -1.0, 0.5, -0.025, 0.1) |K|,
    with the [111] axis normal to the honeycomb plane.
    """
    assert lat.name == "honeycomb"
    idx = {"x": 0, "y": 1, "z": 2}
    H = Operator()
    for b in lat.bonds(1):
        gi = idx[b.kind]
        a, bb = [c for c in range(3) if c != gi]
        M = np.eye(3) * J
        M[gi, gi] += K
        M[a, bb] += Gamma
        M[bb, a] += Gamma
        for c in (a, bb):
            M[gi, c] += Gamma_p
            M[c, gi] += Gamma_p
        H = H + exchange(b.i, b.j, M)
    if J3:
        for b in lat.bonds(3):
            H = H + exchange(b.i, b.j, J3)
    if np.any(np.asarray(h) != 0):
        H = H + zeeman(lat.N, h, g)
    return H


# --------------------------------------------------------------------------- #
#                               toric code                                    #
# --------------------------------------------------------------------------- #
class ToricCodeLattice:
    """Qubits on the edges of an L1 x L2 periodic square lattice.

    Edge index: horizontal edge leaving vertex (x, y) to the right is
    ``2*(x*L2 + y)``, the vertical edge going up is ``2*(x*L2 + y) + 1``.
    """

    def __init__(self, L1: int, L2: Optional[int] = None):
        self.L1, self.L2 = L1, (L1 if L2 is None else L2)
        self.N = 2 * self.L1 * self.L2

    def h(self, x, y):
        return 2 * ((x % self.L1) * self.L2 + (y % self.L2))

    def v(self, x, y):
        return self.h(x, y) + 1

    def star(self, x, y):
        """Edges touching vertex (x, y)."""
        return [self.h(x, y), self.h(x - 1, y), self.v(x, y), self.v(x, y - 1)]

    def plaquette(self, x, y):
        """Edges around the plaquette whose lower-left corner is (x, y)."""
        return [self.h(x, y), self.h(x, y + 1), self.v(x, y), self.v(x + 1, y)]

    def edge_midpoint(self, e):
        c, t = divmod(e, 2)
        x, y = divmod(c, self.L2)
        return np.array([x + 0.5, y]) if t == 0 else np.array([x, y + 0.5])

    def stars(self):
        return [self.star(x, y) for x in range(self.L1) for y in range(self.L2)]

    def plaquettes(self):
        return [self.plaquette(x, y) for x in range(self.L1) for y in range(self.L2)]

    def A(self, x, y) -> Operator:
        return pauli_string([(e, "x") for e in self.star(x, y)])

    def B(self, x, y) -> Operator:
        return pauli_string([(e, "z") for e in self.plaquette(x, y)])

    def logical_Z(self, direction=0) -> Operator:
        """Z string on a non-contractible loop: horizontal edges along x (dir 0)
        or vertical edges along y (dir 1)."""
        if direction == 0:
            return pauli_string([(self.h(x, 0), "z") for x in range(self.L1)])
        return pauli_string([(self.v(0, y), "z") for y in range(self.L2)])

    def logical_X(self, direction=0) -> Operator:
        """X string on the dual lattice (anticommutes with logical_Z(direction))."""
        if direction == 0:
            return pauli_string([(self.h(0, y), "x") for y in range(self.L2)])
        return pauli_string([(self.v(x, 0), "x") for x in range(self.L1)])

    def __repr__(self):
        return f"<ToricCodeLattice {self.L1}x{self.L2}: {self.N} qubits>"


def toric_code(L1: int, L2: Optional[int] = None, Je: float = 1.0, Jm: float = 1.0,
               hx: float = 0.0, hz: float = 0.0, hy: float = 0.0):
    """Kitaev toric code  H = -Je sum_s A_s - Jm sum_p B_p - h . sum_e sigma_e.

    Returns (ToricCodeLattice, Operator).  At h = 0: E0 = -(Je + Jm) L1 L2,
    4-fold degenerate on the torus, gap 2Je (pair of e) / 2Jm (pair of m)
    *per* violated stabiliser, i.e. 4Je for the lowest e-pair.
    """
    tc = ToricCodeLattice(L1, L2)
    H = Operator()
    for x in range(tc.L1):
        for y in range(tc.L2):
            H = H - Je * tc.A(x, y) - Jm * tc.B(x, y)
    for e in range(tc.N):
        if hx:
            H = H - hx * sigma(e, "x")
        if hy:
            H = H - hy * sigma(e, "y")
        if hz:
            H = H - hz * sigma(e, "z")
    return tc, H
