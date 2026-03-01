#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 18:16:32 2025
@author: ritambasu

Optimized, multiprocessing-safe version of radiation_wigner_fast.py
- vectorized rho construction
- fast trace evaluation without building full Akk
- pool initializer to avoid repeatedly pickling rho
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import sympy
from multiprocessing import Pool, cpu_count
import time

# ------------------------ Config ------------------------
# Number of worker processes to use (tweak: 1..cpu_count()).
# Set to 1 while debugging; increase to cpu_count() for heavy runs.
NUM_PROCS = 1

blackhole = 80
data = [[1, 1], [2, 1]]
radiation = 150

# Global read-only holder for rho inside workers (set via initializer)
_RHO = None
_K = None

def psi_row_normalized_matrix(c):
    """Normalize each row of c (shape k x blackhole). Returns s (k x blackhole)."""
    norms = np.linalg.norm(c, axis=1, keepdims=True)
    # Avoid divide-by-zero
    norms[norms == 0] = 1.0
    return c / norms

def build_rho_radiation(k, c, blackhole):
    """
    Vectorized construction of radiation density matrix rho (k x k).
    According to your original definition:
        rho[j, i] = (1/k) * vdot( s_i, s_j )  where s_i are normalized rows (length blackhole)
    This implementation computes that matrix without Python loops.

    NOTE (normalization vs main.tex): in main.tex the reduced radiation state is
      rho_R = (1/(D e^{S0} Z1)) * sum_{i,j} <psi_i|psi_j> |j><i|.
    Here we work with a simplified/normalized overlap model where rho is built from normalized
    random rows s_i and the prefactor is (1/k). If you compare directly to analytic formulas for
    negativity, ensure that the overall normalization conventions match.
    """
    # c is (k, blackhole); normalize rows
    s = psi_row_normalized_matrix(c)  # shape (k, blackhole)
    # rho[j,i] = (1/k) * vdot(s_i, s_j) = (1/k) * sum_b conj(s_j[b]) * s_i[b]
    # Matrix product s.conj() @ s.T gives exactly vdot(s_j, s_i) in entry (j,i).
    rho = (1.0 / k) * (np.conjugate(s) @ s.T)
    return rho

def fast_trace_Akk_rho(a1, a2, rho, k):
    """
    Compute trace(Akk(a1,a2) @ rho) without forming Akk.
    From the definition used in your code, Akk has nonzero entries at positions (l, lp)
    where lp = (2*a1 - l) mod k, and the element is
        exp(2j * pi * a2 * (l - lp) / k)
    Using lp = (2*a1 - l) mod k we can compute l - lp = (2*l - 2*a1) mod k.
    So trace = sum_l phase(l) * rho[ lp(l) , l ].
    """
    # precompute indices and phases
    l = np.arange(k)
    lp = (2 * a1 - l) % k
    # (l - lp) modulo k is equivalent to (2*l - 2*a1) % k, but we can use integer arithmetic
    # Use complex exponential:
    exponent = (2.0 * l - 2.0 * a1) * a2 / k  # real array
    phases = np.exp(2j * np.pi * exponent)   # shape (k,)
    # pick elements rho[lp, l] and sum
    vals = rho[lp, l]
    tr = np.sum(phases * vals)
    return tr

def worker_for_i(i):
    """
    Worker computes sum_{j=0..k-1} |trace(Akk(i,j) @ rho)| / k
    Uses global _RHO and _K set in pool initializer.

    NOTE (meaning): summing over i and j yields sum_{q,p} |W(q,p)| with W(q,p) = (1/k)Tr(rho A(q,p))
    under the standard discrete phase-point operator definition (see main.tex Eq. for A(q,p)).
    """
    global _RHO, _K
    k = _K
    rho = _RHO
    # For fixed i we compute for all j
    # We can compute trace for each j using fast_trace_Akk_rho
    s = 0.0
    for j in range(k):
        tr = fast_trace_Akk_rho(i, j, rho, k)
        s += np.abs(tr) / k
    return s

def pool_initializer(rho, k):
    """Set global variables inside each worker process to avoid pickling rho per task."""
    global _RHO, _K
    _RHO = rho
    _K = k

def compute_neg_parallel(k, rho, num_procs=NUM_PROCS):
    """Compute total negativity-like quantity in parallel by mapping over i."""
    # Use a Pool, initialize each worker with the (read-only) rho and k.
    with Pool(processes=num_procs, initializer=pool_initializer, initargs=(rho, k)) as pool:
        # map worker_for_i over i=0..k-1
        results = pool.map(worker_for_i, range(k))
    return sum(results)

if __name__ == "__main__":
    t0 = time.time()

    for jj in range(2, radiation):
        k = int(sympy.prime(jj))
        print(f"prime k = {k}")

        # random Gaussian coefficients (k x blackhole)
        c = np.random.normal(loc=0.0, scale=1.0, size=(k, blackhole))

        # build rho (k x k)
        rho_radiation = build_rho_radiation(k, c, blackhole)

        # compute neg (parallel)
        neg = compute_neg_parallel(k, rho_radiation, num_procs=NUM_PROCS)
        data.append([k, np.real(neg)])

    data = np.array(data)
    np.save(f'subadd/radiation_wigner/D={blackhole}.npy', data)
    dt = time.time() - t0
    print("done. elapsed {:.2f}s".format(dt))
