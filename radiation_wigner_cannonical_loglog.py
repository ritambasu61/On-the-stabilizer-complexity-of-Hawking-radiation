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
mu = 10
blackhole = 80


# -----------------------------
# Load  data
# -----------------------------

# NOTE (normalization): this plotting script assumes the numerical data saved by
# radiation_wigner_cannonical.py is directly comparable to the analytic curve
# N(r)=erf(sqrt(r))+exp(-r)/sqrt(pi r) from main.tex Eq.(neg). This requires that the
# Wigner function normalization used in the generator script matches main.tex, where
# W(q,p)=(1/D)Tr(\rho A(q,p)), and that the mapping used there for
# r (e.g., r = z1pbyz22p*blackhole/(2*k)) is the intended identification with r=e^{S_2}/(2D).

'''
beta = 0.01
data = np.load(f'subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical_num.npy')
data_ana = np.load(f'subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical_ana.npy')
plt.loglog(data[:, 0], data[:, 1], '.', color='C0', label=rf"Numerical,$~\beta={beta}$")
plt.loglog(data_ana[:, 0], data_ana[:, 1], '-', color='C1', label=rf"Analyticalcal,$~\beta={beta}$")
'''

beta = 0.5
data = np.load(f"subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical_num.npy")
data_ana = np.load(f"subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical_ana.npy")
# data[:,0] is the radiation dimension D (prime k), data[:,1] is the estimated negativity.
# data_ana is the analytic prediction evaluated using the same inferred r as in the generator.
plt.loglog(data[:, 0], data[:, 1], '.', color='C0', label=rf"Numerical,$~\beta={beta}$")
plt.loglog(data_ana[:, 0], data_ana[:, 1], '-', color='C1', label=rf"Analytical,$~\beta={beta}$")



beta = 1.0
data = np.load(f"subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical_num.npy")
data_ana = np.load(f"subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical_ana.npy")
plt.loglog(data[:, 0], data[:, 1], '.', color='C2', label=rf"Numerical,$~\beta={beta}$")
plt.loglog(data_ana[:, 0], data_ana[:, 1], '-', color='C3', label=rf"Analytical,$~\beta={beta}$")



beta = 1.5
data = np.load(f"subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical_num.npy")
data_ana = np.load(f"subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical_ana.npy")
plt.loglog(data[:, 0], data[:, 1], '.', color='C4', label=rf"Numerical,$~\beta={beta}$")
plt.loglog(data_ana[:, 0], data_ana[:, 1], '-', color='C5', label=rf"Analytical,$~\beta={beta}$")


beta = 2.0
data = np.load(f"subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical_num.npy")
data_ana = np.load(f"subadd/radiation_wigner/cannonical/D={blackhole}beta={beta}mu={mu}cannonical_ana.npy")
plt.loglog(data[:, 0], data[:, 1], '.', color='C6', label=rf"Numerical,$~\beta={beta}$")
plt.loglog(data_ana[:, 0], data_ana[:, 1], '-', color='C7', label=rf"Analytical,$~\beta={beta}$")


# -----------------------------
# Labels and legend
# -----------------------------
plt.legend(loc='upper left', frameon=True, fontsize=12)
plt.xlabel(r"$\log(D)\longrightarrow$", fontsize=15, fontweight='bold')
plt.ylabel(r"$\log(\mathcal{N})\longrightarrow$", fontsize=15, fontweight='bold')
plt.xlim([10**0.65, 10**2.01])
#plt.ylim([1, ymax + 0.0015])
plt.tick_params(axis='y', which='both', labelsize=10)
plt.tick_params(axis='x', which='both', labelsize=10)
plt.grid(True, which="both", ls="--", lw=0.5)

# -----------------------------
# Save
# -----------------------------
outfile = f"subadd/radiation_wigner/cannonical/D={blackhole}mu={mu}_loglog_final3"
plt.savefig(outfile + ".pdf", dpi=500)
#plt.savefig(outfile + ".png", dpi=500)

plt.show()
