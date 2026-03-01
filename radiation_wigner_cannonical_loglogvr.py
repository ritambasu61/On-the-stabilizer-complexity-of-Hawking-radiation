#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 18:29:15 2025

@author: ritambasu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import matplotlib as mpl


plt.figure(figsize=(8, 6), dpi=300)
plt.minorticks_on()
plt.tick_params('both', which='minor', length=3, width=1.0, direction='in', top=True, right=True)
plt.tick_params('both', which='major', length=5, width=1.2, direction='in', top=True, right=True)

ax = plt.gca()
for side in ['top', 'right', 'bottom', 'left']:
    ax.spines[side].set_linewidth(1.5)

# -----------------------------
# Parameters
# -----------------------------
lwd = 2
blackhole = 80


# -----------------------------
# Load  data
# -----------------------------
data0 = np.linspace(0.1, 10.5, 200)
print(data0)

# NOTE (meaning): in this "vr" plot, the x-axis is r (not D). The numerical files loaded below
# are expected to store rows [r, N] where N is the computed negativity-like quantity.
# The analytic curve plotted at the end is N(r)=erf(sqrt(r))+exp(-r)/sqrt(pi r), matching
# main.tex Eq.(neg). This comparison is only valid if the generator script computed r in a way
# consistent with r=e^{S_2}/(2D) (or the appropriate ensemble variant) and used the same Wigner
# normalization convention as main.tex.

mu = 30
beta = 0.1
data = np.load(f"subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical2_num.npy")
data_ana = np.load(f"subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical_ana2.npy")
# data[:,0] is r, data[:,1] is the estimated negativity.
plt.loglog(data[:, 0][1:], data[:, 1][1:], '.', color='C0', label=rf"$~\mu={mu},~\beta={beta}$")
data1 = data[:, 0][1:]
#plt.loglog(data_ana[:, 0], data_ana[:, 1], '-', color='C1', label=rf"Analyticalcal,$~\beta={beta}$")

mu = 40
beta = 0.5
data = np.load(f"subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical2_num.npy")
data_ana = np.load(f"subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical_ana2.npy")
plt.loglog(data[:, 0][1:], data[:, 1][1:], '.', color='C1', label=rf"$~\mu={mu},~\beta={beta}$")
data2 = data[:, 0][1:]
#plt.loglog(data_ana[:, 0], data_ana[:, 1], '-', color='C3', label=rf"Analyticalcal,$~\beta={beta}$")


mu = 50
beta = 1.0
data = np.load(f"subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical2_num.npy")
data_ana = np.load(f"subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical_ana2.npy")
plt.loglog(data[:, 0][1:], data[:, 1][1:], '.', color='C2', label=rf"$~\mu={mu},~\beta={beta}$")
data3 = data[:, 0][1:]
#plt.loglog(data_ana[:, 0], data_ana[:, 1], '-', color='C5', label=rf"Analyticalcal,$~\beta={beta}$")


mu = 55
beta = 1.5
data = np.load(f"subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical2_num.npy")
data_ana = np.load(f"subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical_ana2.npy")
plt.loglog(data[:, 0][1:], data[:, 1][1:], '.', color='C3', label=rf"$~\mu={mu},~\beta={beta}$")
data4 = data[:, 0][1:]


mu = 65
beta = 2.0
data = np.load(f"subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical2_num.npy")
data_ana = np.load(f"subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical_ana2.npy")
plt.loglog(data[:, 0][1:], data[:, 1][1:], '.', color='C4', label=rf"$~\mu={mu},~\beta={beta}$")
data5 = data[:, 0][1:]






r = np.sort(np.hstack([data0, data1, data2, data3, data4, data5]))
print(r)
N = erf(np.sqrt(r)) + np.exp(-r) / np.sqrt(np.pi * r)
plt.loglog(r, N, color='C6',label=r"$N(r) = \mathrm{erf}(\sqrt{r}) + \frac{e^{-r}}{\sqrt{\pi r}}$")



# -----------------------------
# Labels and legend
# -----------------------------
plt.xlabel(r"$\log(r)\longrightarrow$", fontsize=15, fontweight='bold')
plt.ylabel(r"$\log(\mathcal{N})\longrightarrow$", fontsize=15, fontweight='bold')
#plt.xlim([10**1.2, 10**2])
#plt.ylim([1, ymax + 0.0015])
plt.tick_params(axis='y', which='both', labelsize=10)
plt.tick_params(axis='x', which='both', labelsize=10)
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend(loc='upper right',frameon=True, fontsize=14)


# -----------------------------
# Save
# -----------------------------
outfile = f"subadd/radiation_wigner/cannonical/D={blackhole}mu={mu}_loglog_final2"
plt.savefig(outfile + ".pdf", dpi=500)
#plt.savefig(outfile + ".png", dpi=500)

plt.show()


