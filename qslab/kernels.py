"""
Compute kernels for applying a compiled Pauli operator to a state vector.

Three implementations share one data layout (flattened
:class:`~qslab.operators.CompiledOperator`):

* ``apply_numba``  shared-memory parallel "pull" kernel (``numba.prange``).
  Each thread writes only its own output entries, so it needs no atomics,
  and it does not store the Hamiltonian.  This is the default CPU path for
  large Hilbert spaces.
* ``apply_xp``     vectorised "push" kernel written for any array module.
  It runs with NumPy on the CPU and CuPy on the GPU.
* scipy CSR        built by ``CompiledOperator.to_sparse`` for small and
  medium problems.
"""
from __future__ import annotations

import numpy as np

from .parallel import HAVE_NUMBA

__all__ = ["flatten_groups", "apply_numba", "apply_xp", "parity_sign_xp"]


def flatten_groups(groups, is_real):
    fms = np.array(list(groups.keys()), dtype=np.uint64)
    zoff = np.zeros(len(fms) + 1, dtype=np.int64)
    zl, cl = [], []
    for g, fm in enumerate(groups):
        zms, cs = groups[fm]
        zl.append(zms)
        cl.append(cs.real if is_real else cs)
        zoff[g + 1] = zoff[g] + len(zms)
    zms = np.concatenate(zl) if zl else np.zeros(0, np.uint64)
    cs = np.concatenate(cl) if cl else np.zeros(0, float if is_real else complex)
    return fms, zoff, zms.astype(np.uint64), np.ascontiguousarray(cs)


def parity_sign_xp(x, xp):
    """(-1)^popcount(x) for uint64 arrays, written so CuPy can run it too."""
    y = x ^ (x >> xp.uint64(32))
    y = y ^ (y >> xp.uint64(16))
    y = y ^ (y >> xp.uint64(8))
    y = y ^ (y >> xp.uint64(4))
    y = y ^ (y >> xp.uint64(2))
    y = y ^ (y >> xp.uint64(1))
    return 1 - 2 * (y & xp.uint64(1)).astype(xp.int8)


def apply_xp(xp, states, full, fms, zoff, zms, cs, v):
    """Push kernel: out[idx(n ^ fm)] += amp_fm(n) v[n].  numpy or cupy."""
    out = xp.zeros(v.shape, dtype=xp.result_type(v.dtype, cs.dtype))
    dim = states.shape[0]
    fms_h = to_host(fms)
    zoff_h = to_host(zoff)
    for g in range(len(fms_h)):
        amp = xp.zeros(dim, dtype=cs.dtype)
        for t in range(int(zoff_h[g]), int(zoff_h[g + 1])):
            z = zms[t]
            amp += cs[t] * parity_sign_xp(states & z, xp)
        new = states ^ xp.uint64(int(fms_h[g]))
        if full:
            out[new.astype(xp.int64)] += amp * v
        else:
            idx = xp.searchsorted(states, new)
            idx = xp.minimum(idx, dim - 1)
            ok = states[idx] == new
            out[idx[ok]] += (amp * v)[ok]
    return out


def to_host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


if HAVE_NUMBA:
    import numba as nb

    @nb.njit(cache=True, inline="always")
    def _psign(x):
        x ^= x >> np.uint64(32)
        x ^= x >> np.uint64(16)
        x ^= x >> np.uint64(8)
        x ^= x >> np.uint64(4)
        x ^= x >> np.uint64(2)
        x ^= x >> np.uint64(1)
        return 1.0 - 2.0 * float(x & np.uint64(1))

    @nb.njit(cache=True, inline="always")
    def _bsearch(arr, x):
        lo, hi = 0, arr.shape[0] - 1
        while lo <= hi:
            mid = (lo + hi) >> 1
            a = arr[mid]
            if a == x:
                return mid
            if a < x:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

    @nb.njit(cache=True, inline="always")
    def _popc(x):
        c = 0
        while x:
            x &= x - np.uint64(1)
            c += 1
        return c

    @nb.njit(parallel=True, cache=True, fastmath=False)
    def _apply_pull(states, full, fms, zoff, zms, cs, v, out, nlo, off_hi, rank_lo, n_dn):
        dim = states.shape[0]
        G = fms.shape[0]
        mask_lo = (np.uint64(1) << np.uint64(nlo)) - np.uint64(1)
        for p in nb.prange(dim):
            m = states[p]
            acc = out[p] * 0
            for g in range(G):
                n = m ^ fms[g]
                if full:
                    idx = np.int64(n)
                else:
                    # Lin two-table lookup: O(1) index of a fixed-popcount state
                    if _popc(n) != n_dn:
                        continue
                    hi = n >> np.uint64(nlo)
                    o = off_hi[hi]
                    if o < 0:
                        continue
                    idx = o + rank_lo[n & mask_lo]
                amp = cs[zoff[g]] * 0
                for t in range(zoff[g], zoff[g + 1]):
                    amp += cs[t] * _psign(n & zms[t])
                # <m|O|n> = amp_fm(n) since O|n> = sum amp |n^fm>
                acc += amp * v[idx]
            out[p] = acc

    @nb.njit(parallel=True, cache=True)
    def _csr_mv(indptr, indices, data, v, out):
        for r in nb.prange(out.shape[0]):
            acc = out[r] * 0
            for k in range(indptr[r], indptr[r + 1]):
                acc += data[k] * v[indices[k]]
            out[r] = acc

    def csr_matvec(M, v):
        """Multithreaded CSR mat-vec (scipy's is single-threaded)."""
        dt = np.result_type(M.dtype, v.dtype)
        out = np.zeros(M.shape[0], dtype=dt)
        _csr_mv(M.indptr, M.indices, M.data.astype(dt, copy=False),
                np.ascontiguousarray(v, dtype=dt), out)
        return out

    def apply_numba(states, full, fms, zoff, zms, cs, v, tables=None):
        dt = np.result_type(v.dtype, cs.dtype)
        v = np.ascontiguousarray(v, dtype=dt)
        csd = cs.astype(dt)
        out = np.zeros(states.shape[0], dtype=dt)
        if full:
            nlo, off_hi, rank_lo, n_dn = 0, np.zeros(1, np.int64), np.zeros(1, np.int64), 0
        else:
            nlo, off_hi, rank_lo, n_dn = tables
        _apply_pull(states, bool(full), fms, zoff, zms, csd, v, out, nlo, off_hi, rank_lo, n_dn)
        return out
else:  # pragma: no cover
    def apply_numba(*a, **k):
        raise RuntimeError("numba not installed")

    def csr_matvec(M, v):
        return M @ v


def lin_tables(states: np.ndarray, N: int, n_dn: int):
    """Lin's two-table perfect hash for fixed-popcount bases.

    index(s) = off_hi[s >> nlo] + rank_lo[s & (2^nlo - 1)]  (states sorted ascending)
    """
    nlo = N // 2
    nhi = N - nlo
    lo_all = np.arange(2 ** nlo, dtype=np.uint64)
    pc_lo = np.bitwise_count(lo_all).astype(np.int64)
    rank_lo = np.zeros(2 ** nlo, np.int64)
    counts_lo = np.zeros(nlo + 1, np.int64)
    for c in range(nlo + 1):
        sel = np.nonzero(pc_lo == c)[0]
        rank_lo[sel] = np.arange(len(sel))
        counts_lo[c] = len(sel)
    hi_all = np.arange(2 ** nhi, dtype=np.uint64)
    pc_hi = np.bitwise_count(hi_all).astype(np.int64)
    need = n_dn - pc_hi
    valid = (need >= 0) & (need <= nlo)
    cnt = np.where(valid, counts_lo[np.clip(need, 0, nlo)], 0)
    off = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    off_hi = np.where(valid, off, -1).astype(np.int64)
    assert off_hi[-1] + cnt[-1] == len(states) or cnt.sum() == len(states)
    return nlo, off_hi, rank_lo, n_dn
