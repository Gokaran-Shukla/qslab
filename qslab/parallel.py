"""
Parallel execution layer: threads, processes, MPI and GPU.

qslab runs at three levels. Only the first is required.

1. **Single node, shared memory** (default)
   * Numba ``prange`` kernels apply H to a vector on all cores
     (:mod:`qslab.kernels`).
   * Dense linear algebra uses multithreaded BLAS/LAPACK.
   * :func:`pmap` runs independent tasks on a process or thread pool:
     sectors, momenta, FTLM samples, parameter scans, MC replicas.
2. **Multiple nodes (optional, needs mpi4py)**
   Launch with ``mpirun -n 64 python script.py`` or ``srun``.
   :func:`pmap` then splits tasks over the MPI ranks; each rank can use its
   own threads.  Monte Carlo methods (VMC, NQS, Majorana-MC) run independent
   Markov chains on every rank and combine averages with ``allreduce``
   (:func:`allreduce_mean`).
3. **GPU (optional, needs CuPy)**
   ``qslab.set_backend('gpu')``, or ``backend='gpu'`` on a solver, moves H
   application, Lanczos, DMRG tensor contractions and Majorana
   diagonalisation to the device.

Configuration
-------------
Environment variables (read at import):

``QSLAB_NUM_THREADS``  threads per process (default: all cores / MPI ranks per node)
``QSLAB_NUM_JOBS``     workers for :func:`pmap` (default: number of cores)
``QSLAB_BACKEND``      ``cpu`` (default) or ``gpu``
``QSLAB_USE_MPI``      ``1`` to force MPI, ``0`` to disable it. By default MPI is
                       used when more than one rank is present.

or at run time::

    import qslab
    qslab.set_num_threads(16)
    qslab.set_backend("gpu")
    print(qslab.parallel_info())
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Iterable, List, Optional

import numpy as np

__all__ = ["set_num_threads", "get_num_threads", "set_backend", "get_backend",
           "xp_module", "to_numpy", "pmap", "mpi_info", "is_root",
           "allreduce_mean", "allreduce_sum", "bcast", "parallel_info",
           "HAVE_NUMBA", "HAVE_CUPY", "HAVE_MPI", "rank_print"]

# --------------------------------------------------------------------------- #
#                          optional dependencies                              #
# --------------------------------------------------------------------------- #
try:  # Numba: shared-memory kernels
    import numba  # noqa: F401
    HAVE_NUMBA = os.environ.get("QSLAB_DISABLE_NUMBA", "0") != "1"
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

try:  # CuPy: GPU
    import cupy  # noqa: F401
    HAVE_CUPY = bool(cupy.cuda.runtime.getDeviceCount() > 0)
except Exception:
    HAVE_CUPY = False

_MPI = None
_COMM = None
HAVE_MPI = False
_use_mpi_env = os.environ.get("QSLAB_USE_MPI", "auto")
if _use_mpi_env != "0":
    try:
        # only import mpi4py when launched by an MPI launcher (or forced),
        # importing it in a plain run can be slow / print warnings
        _launched = any(k in os.environ for k in (
            "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK", "MPI_LOCALNRANKS",
            "SLURM_STEP_NUM_TASKS", "HYDRA_CONTROL_FD"))
        if _launched or _use_mpi_env == "1":
            from mpi4py import MPI as _MPI  # type: ignore
            _COMM = _MPI.COMM_WORLD
            HAVE_MPI = _COMM.Get_size() > 1 or _use_mpi_env == "1"
    except Exception:  # pragma: no cover
        HAVE_MPI = False


def mpi_info():
    """(rank, size) of the MPI world (0, 1 without MPI)."""
    if HAVE_MPI:
        return _COMM.Get_rank(), _COMM.Get_size()
    return 0, 1


def is_root() -> bool:
    return mpi_info()[0] == 0


def rank_print(*a, **k):
    """print() on rank 0 only."""
    if is_root():
        print(*a, **k)


# --------------------------------------------------------------------------- #
#                                threads                                      #
# --------------------------------------------------------------------------- #
def _default_threads() -> int:
    n = os.environ.get("QSLAB_NUM_THREADS")
    if n:
        return max(1, int(n))
    cores = os.cpu_count() or 1
    try:
        cores = len(os.sched_getaffinity(0))
    except Exception:
        pass
    # with several MPI ranks on one node, share the cores among them
    local = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE",
                               os.environ.get("MPI_LOCALNRANKS",
                                              os.environ.get("SLURM_NTASKS_PER_NODE", "1")).split("(")[0]))
    return max(1, cores // max(local, 1))


_NUM_THREADS = _default_threads()
_BACKEND = os.environ.get("QSLAB_BACKEND", "cpu").lower()


def set_num_threads(n: int):
    """Threads used by Numba kernels and BLAS in this process."""
    global _NUM_THREADS
    _NUM_THREADS = max(1, int(n))
    if HAVE_NUMBA:
        import numba
        numba.set_num_threads(min(_NUM_THREADS, numba.config.NUMBA_NUM_THREADS))
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(_NUM_THREADS)
    except Exception:
        pass


def get_num_threads() -> int:
    return _NUM_THREADS


# --------------------------------------------------------------------------- #
#                                 backend                                     #
# --------------------------------------------------------------------------- #
def set_backend(name: str):
    """'cpu' (NumPy/Numba) or 'gpu' (CuPy)."""
    global _BACKEND
    name = name.lower()
    if name not in ("cpu", "gpu"):
        raise ValueError("backend must be 'cpu' or 'gpu'")
    if name == "gpu" and not HAVE_CUPY:
        raise RuntimeError("GPU backend requested but CuPy / a CUDA device is not available. "
                           "Install cupy-cuda12x (or matching) on a GPU node.")
    _BACKEND = name


def get_backend() -> str:
    return _BACKEND


def xp_module(backend: Optional[str] = None):
    """Array module (numpy or cupy) for a backend name."""
    b = (backend or _BACKEND).lower()
    if b == "gpu":
        if not HAVE_CUPY:
            raise RuntimeError("CuPy not available")
        import cupy
        return cupy
    return np


def to_numpy(a):
    """Bring an array back to host memory (no-op for numpy arrays)."""
    if hasattr(a, "get"):
        return a.get()
    return np.asarray(a)


# --------------------------------------------------------------------------- #
#                         task parallelism: pmap                              #
# --------------------------------------------------------------------------- #
def _default_jobs() -> int:
    n = os.environ.get("QSLAB_NUM_JOBS")
    if n:
        return max(1, int(n))
    return max(1, os.cpu_count() or 1)


def _run_chunk(func, chunk):
    return [(i, func(x)) for i, x in chunk]


# with the 'fork' start method workers inherit these globals, so lambdas,
# closures and bound methods work without pickling
_FORK_FUNC = None
_FORK_ITEMS = None


def _call_index(i):
    return _FORK_FUNC(_FORK_ITEMS[i])


def pmap(func: Callable[[Any], Any], items: Iterable[Any], n_jobs: Optional[int] = None,
         backend: str = "auto", threads_per_job: Optional[int] = None) -> List[Any]:
    """Parallel ``[func(x) for x in items]`` with results in input order.

    backend
        ``'auto'``    MPI ranks if running under MPI, else a process pool.
        ``'mpi'``     round-robin over MPI ranks; every rank receives the full
                      result list (allgather).
        ``'process'`` ``ProcessPoolExecutor`` (func must be picklable,
                      i.e. defined at module level).
        ``'thread'``  ``ThreadPoolExecutor``. Good when func spends its time
                      in NumPy/BLAS/Numba code that releases the GIL.
        ``'serial'``  plain loop (debugging).

    Under MPI with ``backend='auto'``, every rank must call pmap with the
    same items.
    """
    items = list(items)
    if not items:
        return []
    if backend == "auto":
        backend = "mpi" if HAVE_MPI and mpi_info()[1] > 1 else "process"
    n_jobs = n_jobs or _default_jobs()
    if backend == "serial" or (backend in ("process", "thread") and (n_jobs == 1 or len(items) == 1)):
        return [func(x) for x in items]

    if backend == "mpi":
        rank, size = mpi_info()
        mine = [(i, x) for i, x in enumerate(items) if i % size == rank]
        local = _run_chunk(func, mine)
        gathered = _COMM.allgather(local) if HAVE_MPI else [local]
        res = [None] * len(items)
        for part in gathered:
            for i, r in part:
                res[i] = r
        return res

    if backend == "thread":
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            return list(ex.map(func, items))

    if backend == "process":
        import multiprocessing as mp
        global _FORK_FUNC, _FORK_ITEMS
        tpj = threads_per_job or max(1, get_num_threads() // n_jobs)
        if hasattr(os, "fork"):
            ctx = mp.get_context("fork")
            _FORK_FUNC, _FORK_ITEMS = func, items
            try:
                with ProcessPoolExecutor(max_workers=min(n_jobs, len(items)), mp_context=ctx,
                                         initializer=set_num_threads, initargs=(tpj,)) as ex:
                    return list(ex.map(_call_index, range(len(items))))
            finally:
                _FORK_FUNC = _FORK_ITEMS = None
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(items)),
                                 initializer=set_num_threads, initargs=(tpj,)) as ex:
            return list(ex.map(func, items))
    raise ValueError(f"unknown backend {backend!r}")


# --------------------------------------------------------------------------- #
#                         MPI reductions (no-ops serially)                    #
# --------------------------------------------------------------------------- #
def allreduce_sum(x):
    """Sum a scalar or numpy array over MPI ranks."""
    if not HAVE_MPI:
        return x
    arr = np.asarray(x)
    out = np.zeros_like(arr)
    _COMM.Allreduce(np.ascontiguousarray(arr), out, op=_MPI.SUM)
    return out if np.ndim(x) else out.item()


def allreduce_mean(x):
    """Average over MPI ranks."""
    if not HAVE_MPI:
        return x
    return allreduce_sum(x) / mpi_info()[1]


def bcast(obj, root: int = 0):
    if not HAVE_MPI:
        return obj
    return _COMM.bcast(obj, root=root)


def parallel_info() -> dict:
    """Summary of the parallel environment."""
    rank, size = mpi_info()
    info = {
        "threads_per_process": get_num_threads(),
        "numba": HAVE_NUMBA,
        "backend": _BACKEND,
        "cupy_gpu": HAVE_CUPY,
        "mpi": HAVE_MPI,
        "mpi_rank": rank,
        "mpi_size": size,
        "cpu_count": os.cpu_count(),
    }
    if HAVE_NUMBA:
        import numba
        info["numba_threads"] = numba.get_num_threads()
    return info


# apply defaults at import
set_num_threads(_NUM_THREADS)


def _cli_info():  # entry point: qslab-info
    import json
    print(json.dumps(parallel_info(), indent=2))
