#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 18:16:32 2025

@author: ritambasu
"""

# radiation_wigner_fast.py
# Faster & multiprocessing-safe version with vectorized Wigner kernel

# --- Set BLAS thread limits BEFORE importing numpy/scipy ---
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from numpy import linalg as LA
from scipy.optimize import fsolve
from scipy.special import gamma
import scipy as sp
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import sympy
import time
from scipy.special import erf

# ------------------------ Config ------------------------
# You can tweak this. Start with 1–4; larger may not help due to overhead.
NUM_PROCS = min(1, cpu_count() or 1)

# ------------------------ Physics/Model Params ------------------------
blackhole = 80
beta = 3.0
mu = 70
radinitial = 3
radiation = 27
iteration = 5 # number of stochastic iterations per k (prime size)

# output collector
data = [[1, 1]]
data_ana =[[1,1]]
error = []

# ------------------------ Helpers ------------------------
def Generate_GUE(n, rng=None):
    """Creates n x n GUE with variance 1/n (Hermitian)."""
    if rng is None:
        rng = np.random.default_rng()
    # real/imag entries ~ N(0, 0.5 / sqrt(n)) to get Wigner scaling
    scale = 0.5 / np.sqrt(n)
    Lambda_real = rng.normal(loc=0.0, scale=scale, size=(n, n))
    Lambda_im   = rng.normal(loc=0.0, scale=scale, size=(n, n))
    Lambda = Lambda_real + 1j * Lambda_im
    # Hermitian
    G = (Lambda + Lambda.T.conj()) / 2.0
    return G

alpha = np.e**12 / (16 * np.pi**2)

def f(x):
    # x expected in [-1, 1]; be mindful of domain
    return (8 * np.pi * alpha * (0.5 * x * np.sqrt(1 - x**2) + (np.pi / 2) - np.arctan(np.sqrt(1 - x**2) / (1 + x))))

def g(x):
    rt = np.sqrt(x)
    return ((-1 / (2 * np.pi**2) * np.sinh(2 * np.pi * rt)) + ((rt / np.pi) * np.cosh(2 * np.pi * rt)))

# ------------------------ Multiprocessing: worker globals ------------------------
_KDIM = None
_RHO  = None

def _init_worker(kdim_in, rho_in):
    """Initializer: set globals once per worker (spawn-safe)."""
    global _KDIM, _RHO
    _KDIM = int(kdim_in)
    _RHO  = np.asarray(rho_in, dtype=np.complex128, order="C")

def _wigner_entry_vectorized(i, j):
    """
    Compute w(i,j) = |Tr(A_{i,j} @ rho)| / k using identity basis simplification.
    With radstate = I_k, the constraint (l + lp) % k == (2*i) % k gives lp = (2*i - l) % k.
    Then:
      Tr(A_{i,j} rho) = sum_l exp(4j*pi*j*(l - i)/k) * rho[lp, l]  with lp as above.
    """
    k = _KDIM
    rho = _RHO
    l = np.arange(k)
    lp = (2 * i - l) % k
    phase = np.exp(4j * np.pi * j * (l - i) / k)
    s = np.sum(rho[lp, l] * phase)
    return float(np.abs(s) / k)

def _worker(arg):
    i, j = arg
    return _wigner_entry_vectorized(i, j)

def parallel_wsum(k, rho, num_procs=NUM_PROCS):
    """
    Compute sum_{i,j} w(i,j) with multiprocessing and vectorized per-call kernel.
    """
    args = [(i, j) for i in range(k) for j in range(k)]
    # chunksize heuristic: a few chunks per process
    nchunks_target = 4 * max(1, num_procs)
    chunksize = max(1, len(args) // nchunks_target)

    with Pool(processes=num_procs, initializer=_init_worker, initargs=(k, rho)) as pool:
        res = pool.map(_worker, args, chunksize=chunksize)

    return float(np.real(np.sum(res)))


rng = np.random.default_rng()  # single RNG; use per-iteration if desired


# --- 1) Transform from GUE to 'sss' (via f,g mapping) ---
Hkk = Generate_GUE(blackhole, rng=rng)
eigenval, eigenvec = LA.eig(Hkk)  # eigenvec columns are eigenvectors (NumPy convention)


# Filter near-real eigenvalues and map via f,g
eigenval2_list = []
for a in range(blackhole):
    if abs(np.imag(eigenval[a])) <= 1e-5:
        v = f(np.real(eigenval[a]))

        def fg(x):
            return g(x) - v

        # root-finding; domain of x for your model—kept guess=3.0 as in original
        root = fsolve(fg, 3.0)[0]
        eigenval2_list.append(root)

if len(eigenval2_list) == 0:
    # fall back: if no eigenvalue passed the filter, keep original real parts
    eigenval2_list = [np.real(ev) for ev in eigenval]

eigenval2 = np.diag(np.asarray(eigenval2_list, dtype=np.complex128))
# Recompose Hkk in transformed spectrum basis
Hkk = eigenvec.conj().T @ eigenval2 @ eigenvec
eigenval, eigenvec = LA.eig(Hkk)


# ------------------------ Main ------------------------
if __name__ == "__main__":
    total_t0 = time.time()
    print(f"Using NUM_PROCS = {NUM_PROCS} (cpu_count={cpu_count()})")

    zz = 0.0

    for jj in range(radinitial, radiation):
        k = sympy.prime(jj)   # radiation Hilbert space dimension (prime)
        print("\n------------------------------------")
        print(f"k = prime({jj}) = {k}")

        sss = 0.0
        zzz = 0.0
        begin = time.time()


        for kk in range(iteration):

            # --- 2) Build random coefficients c and psi(i) states ---
            # radstate is identity basis -> simplifies later Wigner kernel
            # (no need to materialize radstate itself)
            c_real = rng.normal(0.0, 1.0, size=(k, blackhole))
            c_imag = rng.normal(0.0, 1.0, size=(k, blackhole))
            c = (c_real + 1j * c_imag) / np.sqrt(2)
            #c = rng.normal(loc=0.0, scale=1.0, size=(k, blackhole))+1j+rng.normal(loc=0.0, scale=1.0, size=(k, blackhole))

            def psi(i):
                # original formula, kept as-is but vectorized over 'a' to the extent possible
                s = np.zeros(blackhole, dtype=np.complex128)
                for a in range(blackhole):
                    # beware sqrt of negative eigenval[a]; you had sqrt(-eigenval[a]) in comment
                    # keeping your original gamma argument:
                    s += (2**(0.5 - mu)
                          * gamma(mu - 0.5 + np.sqrt(-2*eigenval[a]))
                          * np.exp(-beta * eigenval[a] / 2)
                          * c[i, a] * eigenvec[:, a])
                # normalize
                norm = np.vdot(s, s)
                if norm == 0:
                    return s
                return s / np.sqrt(np.abs(norm))

            # --- 3) Build radiation density matrix rho_radiation ---
            # rho = (1/k) * sum_{i,j} <psi(i)|psi(j)> |j><i|
            # with computational basis |i>,|j| (radstate = I)
            psi_cache = [psi(i) for i in range(k)]
            overlaps = np.empty((k, k), dtype=np.complex128)
            for i in range(k):
                for j in range(k):
                    overlaps[i, j] = np.vdot(psi_cache[i], psi_cache[j])
            # rho_{j,i} = (1/k) * overlaps[i,j]
            rho_radiation = (1.0 / k) * overlaps.T.copy(order="C")

            # --- 4) Parallel Wigner sum using vectorized kernel ---
            step_p0 = time.time()
            neg_val = parallel_wsum(k, rho_radiation, num_procs=NUM_PROCS)
            step_p1 = time.time()

            sss += neg_val

            # --- 5) zprime observable ---
            def zprime(n):
                acc = 0.0
                for a in range(blackhole):
                    acc += np.real(2**(1 - 2*mu)
                                    * np.abs(gamma(mu - 0.5 + np.sqrt(-eigenval[a])))**2
                                    * np.exp(-beta * eigenval[a] ))**n
                return acc / blackhole

            zzz += (zprime(1)**2 / zprime(2)) / iteration
            
            # -----------------------------
            # Analytical function
            # -----------------------------
        z1pbyz22p = zzz
        r = z1pbyz22p*blackhole/(2*k)
        negg = erf(np.sqrt(r))+np.exp(-r)/np.sqrt(np.pi*r)

        data.append([r, sss / iteration])
        data_ana.append([r, negg])
        error.append([r,np.abs((negg-sss/iteration)/(sss/iteration))])
        print([r, sss / iteration])
        print([r, negg])
        end = time.time()
        print(f"time for k={k} is {end - begin:.3f} s")

    print("data =", data)
    print("data_ana =", data_ana)
    data = np.array(data, dtype=float)
    data_ana = np.array(data_ana, dtype=float)
    #os.makedirs('subadd/radiation_wigner/cannonical', exist_ok=True)
    np.save(f'subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical2_num.npy', data)
    np.save(f'subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical_ana2.npy', data_ana)
    #np.save(f'subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical_error2.npy', data_err)

    plt.loglog(data[:, 0], data[:, 1], '.', color='C0', label="Numerical")
    plt.loglog(data_ana[:, 0], data_ana[:, 1], '-', color='C1', label="Analytical")
    #plt.xlim([10**0.25, 10**2.01])
    #plt.show()
    total_t1 = time.time()
    print(f"\nTotal runtime: {total_t1 - total_t0:.3f} s")
