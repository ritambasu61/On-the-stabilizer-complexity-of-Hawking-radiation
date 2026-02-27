import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import erf

'''mpl.use('Agg')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '20'
from matplotlib import cm
from matplotlib.axes import Axes as ax
import sys'''


# Reset to default math font
mpl.rcParams['mathtext.fontset'] = 'dejavusans'   # default
mpl.rcParams['font.family'] = 'sans-serif'        # default





plt.figure(figsize=(8,6),dpi=500)
plt.minorticks_on()
plt.tick_params('both', which='minor', length=3, width=1.0, direction='in', top=False, right=False)
plt.tick_params('both', which='major', length=3, width=1.0, direction='in', top=False, right=False)
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)


# Number of distinct colors you want
num_colors = 15

# Create a custom list of distinct colors based on the 'cubehelix'
cls = [plt.cm.cubehelix(i / (num_colors + 1)) for i in range(num_colors)]
#cls = ['C0', 'C1','C2', 'C3', 'C4']



lwd=2
nn=1


blackhole=80
x = np.load(f'subadd/radiation_wigner/D={blackhole}.npy')
plt.loglog(x[0:, 0], x[0:, 1],'.', label=r'$Numerical$', color=cls[0], lw=lwd)
ymax=np.max(x[0:,1])
print(x)


def f(k):
	return(1+np.sqrt(1+((8*k)/blackhole)))**2
def ff(k):
	return 2*blackhole*k*(np.sqrt(f(k))-1)/np.sqrt(f(k))
def g(k):
	return( blackhole*(-(np.sqrt(f(k))/(2*k))+(f(k)/(8*k))) )
def neg(k):
	return(   1+(8*k**2*(1/np.sqrt(2*np.pi*ff(k)))*np.exp(g(k))/(blackhole*f(k)))  )
def negs(k):
	return( ((2*k/(np.pi*blackhole))**0.5*np.exp(-blackhole/(2*k)))+   erf(np.sqrt(blackhole/(2*k)))  )

x=np.arange(1,10**3,50)
plt.loglog(x,((2*x)/(np.pi*blackhole))**0.5,'.-', label=r'$\sqrt{\frac{2 D}{\pi e^{S_0}}}$', color='r', lw=lwd)
x=np.arange(1,10**3,0.5)
plt.loglog(x,negs(x), label=r'$Analytical$', color='purple', lw=lwd)




plt.legend(loc='upper left',frameon=True,fontsize = 18)
plt.xlabel(r'$\log(\mathrm{D})\longrightarrow$',fontsize=15,fontweight='bold') 
plt.ylabel(r'$\log(\mathcal{N})\longrightarrow$',fontsize=15,fontweight='bold')
plt.xlim([10**0.5,10**3])
plt.ylim([1,ymax+0.015])
plt.tick_params(axis = 'y', which = 'both', labelsize = 10.0)
plt.tick_params(axis = 'x', which = 'both', labelsize = 10.0)
plt.grid(True,which="both")
#plt.lines(True)
plt.savefig('subadd/radiation_wigner/microcannonical_D=' + str(blackhole) + 'loglog2.pdf',)