"""
Stabilizer formalism: exact entanglement of toric-code-type Z2 spin liquids.

A stabilizer state on n qubits is fixed by n independent commuting Pauli
generators.  In the binary symplectic representation a Pauli string is a
row ``(x | z)`` over GF(2).  For a region A::

    S_A / ln 2 = rank_GF2( G restricted to the columns of A ) - |A|

so entanglement entropies of thousands of qubits are computed exactly.
This gives the topological entanglement entropy gamma = ln 2 of the toric
code via the Kitaev-Preskill and Levin-Wen constructions.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .models import ToricCodeLattice

__all__ = ["gf2_rank", "StabilizerState", "toric_code_ground_state",
           "kitaev_preskill_regions", "levin_wen_regions", "topological_entropy"]


def gf2_rank(M: np.ndarray) -> int:
    """Rank over GF(2) (bit-packed Gaussian elimination)."""
    M = (np.asarray(M) % 2).astype(np.uint8)
    if M.size == 0:
        return 0
    rows, cols = M.shape
    packed = np.packbits(M, axis=1)               # rows of bytes
    r = 0
    for c in range(cols):
        byte, bit = divmod(c, 8)
        mask = np.uint8(1 << (7 - bit))
        piv = None
        for i in range(r, rows):
            if packed[i, byte] & mask:
                piv = i
                break
        if piv is None:
            continue
        if piv != r:
            packed[[r, piv]] = packed[[piv, r]]
        hit = (packed[:, byte] & mask).astype(bool)
        hit[r] = False
        packed[hit] ^= packed[r]
        r += 1
        if r == rows:
            break
    return r


class StabilizerState:
    """Pure stabilizer state given by Pauli generators.

    Parameters
    ----------
    n : number of qubits
    generators : list of Pauli strings, each a list of (qubit, 'x'|'y'|'z')
    """

    def __init__(self, n: int, generators: Sequence[Sequence[Tuple[int, str]]]):
        self.n = n
        G = np.zeros((len(generators), 2 * n), np.uint8)
        for r, g in enumerate(generators):
            for q, a in g:
                if a in "xy":
                    G[r, q] = 1
                if a in "yz":
                    G[r, n + q] = 1
        self.G = G
        rk = gf2_rank(G)
        if rk != n:
            raise ValueError(f"generators have rank {rk}, need {n} for a pure state "
                             "(remove dependent ones / add logical operators)")
        if not self._commuting():
            raise ValueError("generators do not commute")

    def _commuting(self) -> bool:
        n = self.n
        X, Z = self.G[:, :n].astype(int), self.G[:, n:].astype(int)
        S = (X @ Z.T + Z @ X.T) % 2
        return not S.any()

    def entropy(self, A: Iterable[int]) -> float:
        """Entanglement entropy of region A in units of ln 2."""
        A = sorted(set(A))
        if not A or len(A) == self.n:
            return 0.0
        cols = A + [self.n + a for a in A]
        return float(gf2_rank(self.G[:, cols]) - len(A))


def toric_code_ground_state(L1: int, L2: int = None, sector=(1, 1)) -> Tuple[ToricCodeLattice, StabilizerState]:
    """Toric-code ground state as a stabilizer state.

    Generators: all stars and plaquettes except one of each (they multiply to
    the identity) plus the two logical Z operators fixing the sector.
    """
    tc = ToricCodeLattice(L1, L2)
    gens = []
    stars = tc.stars()
    plaqs = tc.plaquettes()
    for s in stars[:-1]:
        gens.append([(e, "x") for e in s])
    for p in plaqs[:-1]:
        gens.append([(e, "z") for e in p])
    gens.append([(tc.h(x, 0), "z") for x in range(tc.L1)])
    gens.append([(tc.v(0, y), "z") for y in range(tc.L2)])
    return tc, StabilizerState(tc.N, gens)


def _edges_in(tc: ToricCodeLattice, pred) -> List[int]:
    return [e for e in range(tc.N) if pred(tc.edge_midpoint(e))]


def kitaev_preskill_regions(tc: ToricCodeLattice, R: float = None, center=None):
    """Three 120-degree sectors A, B, C of a disk (Kitaev-Preskill 2006)."""
    L = min(tc.L1, tc.L2)
    R = R if R is not None else 0.35 * L
    c = np.array(center if center is not None else (tc.L1 / 2, tc.L2 / 2)) + 0.13

    def sector(k):
        def pred(p):
            d = p - c
            if np.hypot(*d) > R:
                return False
            ang = (np.degrees(np.arctan2(d[1], d[0])) + 360) % 360
            return 120 * k <= ang < 120 * (k + 1)
        return pred
    return [_edges_in(tc, sector(k)) for k in range(3)]


def levin_wen_regions(tc: ToricCodeLattice, r_in: float = None, r_out: float = None):
    """Levin-Wen (2006) annulus split into four arcs A (left), B (top),
    C (bottom), D (right)::

        -2 gamma = (S_ABCD - S_ACD) - (S_ABD - S_AD)

    i.e. full annulus minus annulus cut once, minus (cut once minus cut twice).
    """
    L = min(tc.L1, tc.L2)
    r_in = r_in if r_in is not None else 0.15 * L
    r_out = r_out if r_out is not None else 0.42 * L
    c = np.array((tc.L1 / 2, tc.L2 / 2)) + 0.13

    def region(lo, hi):
        def pred(p):
            d = p - c
            r = np.hypot(*d)
            if not (r_in <= r <= r_out):
                return False
            a = (np.degrees(np.arctan2(d[1], d[0])) + 360) % 360
            return (lo <= a < hi) if lo < hi else (a >= lo or a < hi)
        return pred
    return (_edges_in(tc, region(135, 225)), _edges_in(tc, region(45, 135)),
            _edges_in(tc, region(225, 315)), _edges_in(tc, region(315, 45)))


def topological_entropy(state: StabilizerState, A, B, C, D=None, scheme: str = "KP") -> float:
    """Topological entanglement entropy gamma in nats (toric code: ln 2).

    scheme 'KP'  : -gamma = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC
    scheme 'LW'  : -2 gamma = (S_ABCD - S_ACD) - (S_ABD - S_AD) with the four
                   annulus arcs from :func:`levin_wen_regions`.
    """
    S = lambda *R: state.entropy(sorted(set().union(*map(set, R))))
    if scheme == "KP":
        val = S(A) + S(B) + S(C) - S(A, B) - S(B, C) - S(A, C) + S(A, B, C)
        return float(-val * np.log(2))
    if scheme == "LW":
        val = (S(A, B, C, D) - S(A, C, D)) - (S(A, B, D) - S(A, D))
        return float(-val * np.log(2) / 2)
    raise ValueError(scheme)
