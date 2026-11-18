"""
Lattices for frustrated magnetism.

A :class:`Lattice` is a finite cluster cut out of an infinite 2D Bravais
lattice with a multi-site basis.  Clusters can be

* rectangular ``L1 x L2`` (in units of the primitive vectors), with periodic
  or open boundaries independently in each direction (torus, cylinder, strip);
* general periodic supercells given by an integer matrix ``M`` whose rows
  are the cluster vectors ``T_i = M_i1 a1 + M_i2 a2`` (tilted ED clusters
  such as the 36-site kagome or the sqrt(12) x sqrt(12) honeycomb).

Everything downstream (ED, DMRG, Majorana solver, partons, NQS) consumes the
same objects:

``lat.bonds(k)``        k-th neighbour bonds as :class:`Bond` records
``lat.plaquettes(kind)`` elementary loops (hexagons, triangles, squares)
``lat.translations()``  site permutations for a1 / a2 translations
``lat.momenta()``       allowed crystal momenta of the cluster

Site ordering is ``index = (cell_index) * n_basis + sublattice`` where, for
rectangular clusters, ``cell_index = x1 * L2 + x2``.  For a cylinder that is
open along a1 and periodic along a2 this is column-by-column ordering, which
is the standard snake used for DMRG on cylinders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SQ3 = np.sqrt(3.0)

__all__ = [
    "Bond", "Plaquette", "Lattice",
    "chain", "square", "triangular", "honeycomb", "kagome",
    "make_lattice",
]


@dataclass(frozen=True)
class Bond:
    """An (undirected) bond of the cluster.

    Attributes
    ----------
    i, j : int
        Site indices.  For honeycomb lattices ``i`` is always on sublattice A.
    d : tuple(float, float)
        Real-space vector from ``i`` to ``j`` in the *infinite* lattice.
    order : int
        Neighbour shell (1 = nearest neighbour, 2 = next-nearest, ...).
    wind : tuple(int, int)
        How many times the bond wraps around the periodic boundary along the
        cluster vectors T1, T2 (signed).  Used for twisted boundary
        conditions, flux insertion and topological sectors.
    kind : str
        Bond label.  For the honeycomb nearest-neighbour bonds this is the
        Kitaev label ``'x'``, ``'y'`` or ``'z'``; otherwise ``''``.
    """

    i: int
    j: int
    d: Tuple[float, float]
    order: int
    wind: Tuple[int, int] = (0, 0)
    kind: str = ""

    @property
    def wraps(self) -> bool:
        return self.wind != (0, 0)


@dataclass(frozen=True)
class Plaquette:
    """An elementary loop, sites ordered counter-clockwise."""

    sites: Tuple[int, ...]
    center: Tuple[float, float]
    kind: str
    # real-space positions of the loop vertices (unwrapped, consistent with d)
    coords: Tuple[Tuple[float, float], ...] = field(default=(), compare=False)


# --------------------------------------------------------------------------- #
#                                Lattice class                                #
# --------------------------------------------------------------------------- #
class Lattice:
    """Finite cluster of a 2D lattice with a basis.

    Parameters
    ----------
    name : str
    a1, a2 : array_like (2,)
        Primitive vectors.
    basis : array_like (nb, 2)
        Cartesian positions of the basis sites inside the unit cell.
    L1, L2 : int
        Cluster size (rectangular cluster).  Ignored if ``supercell`` given.
    bc : str or (str, str)
        'periodic' / 'open' per direction.  ``'cylinder'`` is shorthand for
        ('open', 'periodic') - open along a1, periodic along a2.
    supercell : (2,2) int array, optional
        Rows are cluster vectors in units of (a1, a2).  Always periodic.
    plaquette_defs : dict, optional
        ``kind -> (list of center offsets, radius)``, used by
        :meth:`plaquettes`.
    """

    def __init__(self, name: str, a1, a2, basis, L1: int = 1, L2: int = 1,
                 bc="periodic", supercell=None,
                 plaquette_defs: Optional[Dict[str, Tuple[list, float]]] = None,
                 bond_labeler=None):
        self.name = name
        self.a1 = np.asarray(a1, float)
        self.a2 = np.asarray(a2, float)
        self.A = np.array([self.a1, self.a2])           # rows = primitive vecs
        self.basis = np.atleast_2d(np.asarray(basis, float))
        self.nb = len(self.basis)
        self._plaq_defs = plaquette_defs or {}
        self._bond_labeler = bond_labeler

        if supercell is not None:
            M = np.asarray(supercell, int)
            if abs(round(np.linalg.det(M))) < 1:
                raise ValueError("singular supercell matrix")
            self.M = M
            self.bc = ("periodic", "periodic")
            self.rectangular = False
            self.L1 = self.L2 = None
        else:
            if isinstance(bc, str):
                if bc == "cylinder":
                    bc = ("open", "periodic")
                else:
                    bc = (bc, bc)
            for b in bc:
                if b not in ("periodic", "open"):
                    raise ValueError(f"unknown boundary condition {b!r}")
            self.bc = tuple(bc)
            self.M = np.array([[L1, 0], [0, L2]], int)
            self.rectangular = True
            self.L1, self.L2 = int(L1), int(L2)

        self.periodic = tuple(b == "periodic" for b in self.bc)
        self.T = self.M @ self.A            # cluster vectors (rows), cartesian
        self._Minv = np.linalg.inv(self.M)

        self.cells = self._enumerate_cells()
        self.ncells = len(self.cells)
        self.N = self.ncells * self.nb
        self._cell_index = {tuple(c): n for n, c in enumerate(self.cells)}

        # positions and sublattice labels
        pos = np.zeros((self.N, 2))
        sub = np.zeros(self.N, int)
        cell_of = np.zeros((self.N, 2), int)
        for n, c in enumerate(self.cells):
            for s in range(self.nb):
                idx = n * self.nb + s
                pos[idx] = c[0] * self.a1 + c[1] * self.a2 + self.basis[s]
                sub[idx] = s
                cell_of[idx] = c
        self.positions = pos
        self.sublattice = sub
        self.cell_of_site = cell_of
        self._bond_cache: Dict[int, List[Bond]] = {}
        self._shells: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def __repr__(self):
        if self.rectangular:
            geo = f"{self.L1}x{self.L2}, bc={self.bc}"
        else:
            geo = f"supercell={self.M.tolist()}"
        return f"<Lattice {self.name}: N={self.N} sites, {geo}>"

    def __len__(self):
        return self.N

    # ------------------------------------------------------------------ #
    #                         cell bookkeeping                           #
    # ------------------------------------------------------------------ #
    def _enumerate_cells(self) -> np.ndarray:
        if self.rectangular:
            return np.array([(x1, x2) for x1 in range(self.L1)
                             for x2 in range(self.L2)], int)
        # general supercell: reduce a bounding box modulo M
        M = self.M
        corners = np.array([[0, 0], M[0], M[1], M[0] + M[1]])
        lo, hi = corners.min(0), corners.max(0)
        seen = {}
        for x1 in range(lo[0] - 1, hi[0] + 2):
            for x2 in range(lo[1] - 1, hi[1] + 2):
                r = self._reduce((x1, x2))
                seen[r] = True
        cells = np.array(sorted(seen.keys()), int)
        assert len(cells) == abs(round(np.linalg.det(M))), "cell enumeration failed"
        return cells

    def _reduce(self, n) -> Tuple[int, int]:
        """Reduce integer cell coordinate into the cluster (periodic dirs)."""
        n = np.asarray(n, int)
        if self.rectangular:
            out = n.copy()
            if self.periodic[0]:
                out[0] %= self.L1
            if self.periodic[1]:
                out[1] %= self.L2
            return int(out[0]), int(out[1])
        f = n @ self._Minv
        w = np.floor(f + 1e-9).astype(int)
        r = n - w @ self.M
        return int(r[0]), int(r[1])

    def _winding(self, n) -> Tuple[int, int]:
        n = np.asarray(n, int)
        if self.rectangular:
            w1 = n[0] // self.L1 if self.periodic[0] else 0
            w2 = n[1] // self.L2 if self.periodic[1] else 0
            return int(w1), int(w2)
        f = n @ self._Minv
        w = np.floor(f + 1e-9).astype(int)
        return int(w[0]), int(w[1])

    def _inside(self, n) -> bool:
        if not self.rectangular:
            return True
        if not self.periodic[0] and not (0 <= n[0] < self.L1):
            return False
        if not self.periodic[1] and not (0 <= n[1] < self.L2):
            return False
        return True

    def site_index(self, cell, s: int) -> Optional[int]:
        """Index of basis site ``s`` in (possibly out-of-cluster) ``cell``."""
        if not self._inside(cell):
            return None
        r = self._reduce(cell)
        return self._cell_index[r] * self.nb + s

    # ------------------------------------------------------------------ #
    #                              bonds                                 #
    # ------------------------------------------------------------------ #
    def neighbor_distances(self, nshell: int = 4) -> np.ndarray:
        """Distances of the first ``nshell`` neighbour shells (infinite lattice)."""
        if self._shells is not None and len(self._shells) >= nshell:
            return self._shells[:nshell]
        R = 4
        ds = []
        for s in range(self.nb):
            for t in range(self.nb):
                for n1 in range(-R, R + 1):
                    for n2 in range(-R, R + 1):
                        d = n1 * self.a1 + n2 * self.a2 + self.basis[t] - self.basis[s]
                        r = np.linalg.norm(d)
                        if r > 1e-8:
                            ds.append(r)
        ds = np.unique(np.round(ds, 8))
        self._shells = ds
        return ds[:nshell]

    def bonds(self, order: int = 1) -> List[Bond]:
        """All bonds of a given neighbour order (1 = NN, 2 = NNN, 3 = NNNN).

        Each bond of the *infinite* lattice that maps into the cluster is
        returned exactly once, so on small periodic clusters two sites can
        be connected by more than one bond (with different ``d``).  This is
        what makes ED energies on tori converge to the thermodynamic limit.
        """
        if order in self._bond_cache:
            return self._bond_cache[order]
        dist = self.neighbor_distances(max(order, 4))[order - 1]
        R = 4
        found = {}
        for c in self.cells:
            for s in range(self.nb):
                i = self.site_index(c, s)
                for n1 in range(-R, R + 1):
                    for n2 in range(-R, R + 1):
                        for t in range(self.nb):
                            d = (n1 * self.a1 + n2 * self.a2
                                 + self.basis[t] - self.basis[s])
                            if abs(np.linalg.norm(d) - dist) > 1e-6:
                                continue
                            cj = (c[0] + n1, c[1] + n2)
                            j = self.site_index(cj, t)
                            if j is None:
                                continue
                            if i == j:
                                continue  # self-bond on a tiny cluster
                            dr = (round(d[0], 8), round(d[1], 8))
                            key_f = (i, j, dr)
                            key_b = (j, i, (-dr[0] + 0.0, -dr[1] + 0.0))
                            key = min(key_f, key_b)
                            if key in found:
                                continue
                            wind = self._winding(cj)
                            if key == key_b:   # store in canonical direction
                                wind = (-wind[0], -wind[1])
                            found[key] = wind
        bl = []
        for (i, j, dr), wind in found.items():
            b = Bond(i, j, dr, order, wind, "")
            if self._bond_labeler is not None:
                b = self._bond_labeler(self, b)
            bl.append(b)
        bl.sort(key=lambda b: (b.i, b.j, b.d))
        self._bond_cache[order] = bl
        return bl

    def nn(self):
        return self.bonds(1)

    def nnn(self):
        return self.bonds(2)

    def neighbors(self, i: int, order: int = 1) -> List[int]:
        out = []
        for b in self.bonds(order):
            if b.i == i:
                out.append(b.j)
            elif b.j == i:
                out.append(b.i)
        return out

    def coordination(self, order: int = 1) -> int:
        return 2 * len(self.bonds(order)) // self.N

    # ------------------------------------------------------------------ #
    #                           plaquettes                               #
    # ------------------------------------------------------------------ #
    def plaquettes(self, kind: Optional[str] = None) -> List[Plaquette]:
        """Elementary loops.  ``kind`` e.g. 'hexagon', 'up', 'down', 'square'.

        Plaquettes that would be cut by an open boundary are dropped.
        """
        kinds = [kind] if kind else list(self._plaq_defs)
        out = []
        for k in kinds:
            if k not in self._plaq_defs:
                raise KeyError(f"lattice {self.name} has no plaquette kind {k!r}; "
                               f"available: {list(self._plaq_defs)}")
            offsets, radius = self._plaq_defs[k]
            for c in self.cells:
                for off in offsets:
                    center = c[0] * self.a1 + c[1] * self.a2 + np.asarray(off)
                    p = self._loop_around(center, radius, k)
                    if p is not None:
                        out.append(p)
        return out

    def _loop_around(self, center, radius, kind) -> Optional[Plaquette]:
        """Collect sites at distance ``radius`` from ``center`` (infinite lattice)."""
        verts = []
        # integer cell of the center
        f = np.linalg.solve(self.A.T, center)
        c0 = np.floor(f).astype(int)
        for n1 in range(c0[0] - 2, c0[0] + 3):
            for n2 in range(c0[1] - 2, c0[1] + 3):
                for s in range(self.nb):
                    p = n1 * self.a1 + n2 * self.a2 + self.basis[s]
                    if abs(np.linalg.norm(p - center) - radius) < 1e-6:
                        idx = self.site_index((n1, n2), s)
                        if idx is None:
                            return None
                        ang = np.arctan2(*(p - center)[::-1])
                        verts.append((ang, idx, tuple(p)))
        if len(verts) < 3:
            return None
        verts.sort()
        return Plaquette(tuple(v[1] for v in verts), tuple(center), kind,
                         tuple(v[2] for v in verts))

    # ------------------------------------------------------------------ #
    #                     translations and momenta                       #
    # ------------------------------------------------------------------ #
    def translation(self, shift=(1, 0)) -> np.ndarray:
        """Permutation ``perm`` with ``site i -> perm[i]`` under translation."""
        if self.rectangular and not all(self.periodic):
            raise ValueError("translations need a fully periodic cluster")
        perm = np.zeros(self.N, int)
        for idx in range(self.N):
            c = self.cell_of_site[idx]
            s = self.sublattice[idx]
            perm[idx] = self.site_index((c[0] + shift[0], c[1] + shift[1]), s)
        return perm

    def translations(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.translation((1, 0)), self.translation((0, 1))

    def reciprocal(self) -> np.ndarray:
        """Primitive reciprocal vectors b1, b2 (rows), a_i . b_j = 2 pi delta_ij."""
        return 2 * np.pi * np.linalg.inv(self.A).T

    def momenta(self) -> np.ndarray:
        """Crystal momenta compatible with the periodic cluster (cartesian)."""
        if self.rectangular and not all(self.periodic):
            raise ValueError("momenta need a fully periodic cluster")
        G = 2 * np.pi * np.linalg.inv(self.T).T     # reciprocal of cluster
        B = self.reciprocal()
        ks = []
        n = self.ncells
        for m1 in range(-n, n + 1):
            for m2 in range(-n, n + 1):
                k = m1 * G[0] + m2 * G[1]
                # reduce into the first reciprocal cell parallelogram
                f = np.linalg.solve(B.T, k)
                f = f - np.floor(f + 1e-9)
                ks.append(tuple(np.round(f, 8)))
        ks = sorted(set(ks))
        assert len(ks) == n, (len(ks), n)
        return np.array([f[0] * B[0] + f[1] * B[1] for f in ks])

    def high_symmetry_points(self) -> Dict[str, np.ndarray]:
        B = self.reciprocal()
        pts = {"G": np.zeros(2)}
        if self.name in ("triangular", "honeycomb", "kagome"):
            pts["M"] = 0.5 * B[0]
            pts["K"] = (2 * B[0] + B[1]) / 3
        elif self.name in ("square",):
            pts["X"] = 0.5 * B[0]
            pts["M"] = 0.5 * (B[0] + B[1])
        else:
            pts["X"] = 0.5 * B[0]
        return pts

    def kpath(self, labels: Sequence[str], n: int = 60) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Straight-line path through high-symmetry points.

        Returns (k points (m,2), cumulative distance, tick positions).
        """
        P = self.high_symmetry_points()
        ks, dist, ticks = [], [], [0.0]
        acc = 0.0
        for a, b in zip(labels[:-1], labels[1:]):
            ka, kb = P[a], P[b]
            seg = np.linspace(0, 1, n, endpoint=False)
            for t in seg:
                k = ka + t * (kb - ka)
                if ks:
                    acc += np.linalg.norm(k - ks[-1])
                ks.append(k)
                dist.append(acc)
            acc += np.linalg.norm(kb - ks[-1])
            ticks.append(acc)
            ks.append(kb)
            dist.append(acc)
            ks.pop(); dist.pop()
        ks.append(P[labels[-1]])
        dist.append(acc)
        return np.array(ks), np.array(dist), ticks

    # ------------------------------------------------------------------ #
    def sublattice_sign(self) -> np.ndarray:
        """(+1/-1) staggering for bipartite lattices (Marshall sign rule)."""
        col = np.zeros(self.N, int) - 1
        col[0] = 0
        stack = [0]
        nbrs = [[] for _ in range(self.N)]
        for b in self.bonds(1):
            nbrs[b.i].append(b.j)
            nbrs[b.j].append(b.i)
        bip = True
        while stack:
            i = stack.pop()
            for j in nbrs[i]:
                if col[j] < 0:
                    col[j] = 1 - col[i]
                    stack.append(j)
                elif col[j] == col[i]:
                    bip = False
        if not bip:
            raise ValueError(f"{self.name} cluster is not bipartite")
        return 1 - 2 * col

    def plot(self, ax=None, orders=(1,), show_index=True, plaquettes=False,
             bond_colors=None):
        """Quick matplotlib drawing of the cluster (wrapping bonds dashed)."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        cols = {1: "k", 2: "tab:blue", 3: "tab:green"}
        kcols = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}
        for o in orders:
            for b in self.bonds(o):
                p = self.positions[b.i]
                q = p + np.array(b.d)
                c = kcols.get(b.kind, cols.get(o, "gray"))
                if bond_colors and b.kind in bond_colors:
                    c = bond_colors[b.kind]
                ax.plot([p[0], q[0]], [p[1], q[1]], color=c,
                        ls="--" if b.wraps else "-", lw=1.2 if o == 1 else 0.7,
                        alpha=0.5 if b.wraps else 1)
        ax.scatter(*self.positions.T, s=60, c=self.sublattice, cmap="tab10",
                   zorder=3, edgecolors="k")
        if show_index:
            for i, p in enumerate(self.positions):
                ax.annotate(str(i), p, xytext=(3, 3), textcoords="offset points",
                            fontsize=7)
        if plaquettes:
            for p in self.plaquettes():
                ax.plot(*p.center, "x", color="gray")
        ax.set_aspect("equal")
        ax.set_title(repr(self))
        return ax


# --------------------------------------------------------------------------- #
#                               lattice presets                               #
# --------------------------------------------------------------------------- #
def _honeycomb_labeler(lat: Lattice, b: Bond) -> Bond:
    """Kitaev x/y/z labels for honeycomb NN bonds, with i on sublattice A."""
    if b.order != 1:
        return b
    i, j, d, w = b.i, b.j, np.array(b.d), b.wind
    if lat.sublattice[i] == 1:            # orient A -> B
        i, j, d, w = j, i, -d, (-w[0], -w[1])
    ang = np.degrees(np.arctan2(d[1], d[0])) % 360
    # A->B vectors: z = (0,-1/sqrt3) [270deg], x = (1/2, sqrt3/6) [30deg],
    # y = (-1/2, sqrt3/6) [150deg]
    if abs(ang - 270) < 1:
        kind = "z"
    elif abs(ang - 30) < 1:
        kind = "x"
    elif abs(ang - 150) < 1:
        kind = "y"
    else:  # pragma: no cover
        raise RuntimeError(f"unexpected honeycomb bond angle {ang}")
    return Bond(i, j, (float(d[0]), float(d[1])), b.order, w, kind)


def chain(L: int, bc="periodic") -> Lattice:
    """1D chain (lattice spacing 1)."""
    return Lattice("chain", (1, 0), (0, 1), [(0, 0)], L, 1,
                   bc=(bc, "open"))


def square(L1: int, L2: Optional[int] = None, bc="periodic", supercell=None) -> Lattice:
    L2 = L1 if L2 is None else L2
    return Lattice("square", (1, 0), (0, 1), [(0, 0)], L1, L2, bc, supercell,
                   plaquette_defs={"square": ([(0.5, 0.5)], np.sqrt(0.5))})


def triangular(L1: int, L2: Optional[int] = None, bc="periodic", supercell=None) -> Lattice:
    L2 = L1 if L2 is None else L2
    a1, a2 = (1, 0), (0.5, SQ3 / 2)
    up = (0.5, SQ3 / 6)
    down = (1.0, SQ3 / 3)
    r = 1 / SQ3
    return Lattice("triangular", a1, a2, [(0, 0)], L1, L2, bc, supercell,
                   plaquette_defs={"up": ([up], r), "down": ([down], r)})


def honeycomb(L1: int, L2: Optional[int] = None, bc="periodic", supercell=None) -> Lattice:
    """Honeycomb lattice, NN distance 1/sqrt(3), A at 0, B at (a1+a2)/3.

    Nearest-neighbour bonds carry Kitaev labels (``bond.kind``) x, y, z.
    """
    L2 = L1 if L2 is None else L2
    a1, a2 = np.array([1.0, 0.0]), np.array([0.5, SQ3 / 2])
    basis = [(0, 0), tuple((a1 + a2) / 3)]
    r = 1 / SQ3
    return Lattice("honeycomb", a1, a2, basis, L1, L2, bc, supercell,
                   plaquette_defs={"hexagon": ([(0.0, r)], r)},
                   bond_labeler=_honeycomb_labeler)


def kagome(L1: int, L2: Optional[int] = None, bc="periodic", supercell=None) -> Lattice:
    """Kagome lattice with NN distance 1 (a1 = (2,0), a2 = (1, sqrt3))."""
    L2 = L1 if L2 is None else L2
    a1, a2 = (2.0, 0.0), (1.0, SQ3)
    basis = [(0, 0), (1, 0), (0.5, SQ3 / 2)]
    r_tri = 1 / SQ3
    return Lattice("kagome", a1, a2, basis, L1, L2, bc, supercell,
                   plaquette_defs={
                       "up": ([(0.5, SQ3 / 6)], r_tri),
                       "down": ([(1.5, -SQ3 / 6)], r_tri),
                       "hexagon": ([(1.5, SQ3 / 2)], 1.0),
                   })


def make_lattice(name: str, *args, **kw) -> Lattice:
    """Factory: ``make_lattice('kagome', 2, 3, bc='cylinder')``."""
    table = {"chain": chain, "square": square, "triangular": triangular,
             "honeycomb": honeycomb, "kagome": kagome}
    if name not in table:
        raise KeyError(f"unknown lattice {name!r}; choose from {list(table)}")
    return table[name](*args, **kw)
