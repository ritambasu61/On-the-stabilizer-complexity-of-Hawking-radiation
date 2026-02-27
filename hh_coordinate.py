import numpy as np
from numpy import linalg as LA
from scipy.linalg import expm
from scipy.sparse import csr_matrix
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as mpl
from multiprocessing import Pool
import time as tim
from qutip import Qobj, ptrace

#initial specifications of the parameters
db = 1
dr = 101
ti = 0.0
tf = 0.5
dt = 0.01
num_processes = 10  # Use  available CPUs
data = []




begin=tim.time()#this is for calculating the time to run the code(can be ignored)


def Generate_GUE(n):
    """Creates nxn GUE"""
    i = complex(0,1)
    Lambda_real = np.random.normal(scale=1/np.sqrt(n),size=[n,n])
    Lambda_im = np.random.normal(scale=1/np.sqrt(n),size=[n,n])
    Lambda = Lambda_real + Lambda_im * i
    G = (Lambda+Lambda.T.conjugate())/2
    return G




# Hamiltonians
#Hb = Generate_GUE(db)
#Hr = Generate_GUE(dr)
#Hkk = np.kron(Hb, Hr)
Hkk = Generate_GUE(db*dr)
np.save('/home/ritam.basu/Desktop/hh_coordinate/hkk_Db=' + str(db) + 'Dr=' + str(dr) + '.npy', Hkk) #just copy-paste the path where you want to save the random Hamiltonian
print('Hamiltonian')



#krylov basis
Kryk = np.identity(dr, dtype=np.complex128)


def Ua(t):
    return expm(-1j * t * Hkk * np.sqrt(dr * db))  # Time evolution operator




# Initial states
psi_b0 = np.zeros(db, dtype=np.complex128) #initial state of black hole
psi_b0[0] = 1
psi_r0 = np.zeros(dr, dtype=np.complex128) #initial state of radiation
psi_r0[0] = 1
rho_b0 = np.outer(psi_b0, psi_b0.conj()) #initial density matrix of blackhole
rho_r0 = np.outer(psi_r0, psi_r0.conj()) #initial density matrix of radiation
rho_0 = np.kron(rho_b0, rho_r0)          #state of the total thermodynamic system





#reduced and time evolved density operator for radiation
def rho_r(t):
    """Computes reduced density matrix using QuTiP."""
    U = Ua(t)
    rho_t = U @ rho_0 @ U.conj().T  # Time-evolved density matrix
    rho_t_q = Qobj(rho_t, dims=[[db, dr], [db, dr]])  # Convert to QuTiP object
    return ptrace(rho_t_q, 1)  # Trace out first subsystem (db)
#print(np.shape(rho_r(1)))




nn = 1    #renni term (can be ignored)
time = np.arange(ti, tf, dt)
# Assuming dr, Kryk, rho_r, nn are defined globally




#phase space translation operator
def Akk(a1, a2, dr, Kryk):
    """Computes the Akk matrix."""
    result = np.zeros((dr, dr), dtype=np.complex128)
    for l in range(dr):
        for lp in range(dr):
            if (l + lp) % dr != (2 * a1) % dr:
                continue
            fact = np.exp(2j * np.pi * (a2 * (l - lp) / dr))
            result += fact * np.outer(Kryk[lp], Kryk[l].conj().T)
    return result





#wigner function
def wsykenkk(args):
    """Computes the function wsykenkk(x, y) for given (x, y)."""
    x, y, tt, dr, Kryk, rho_r_t, nn = args
    A = Akk(x, y, dr, Kryk)
    return (np.abs(np.trace(A @ rho_r_t)) ** nn) / dr





#parallelisation of the whole or equivalently parallelization of wigner function
def parallel_wsykenkk(tt, dr, Kryk, rho_r, nn):
    """Parallel computation of wsykenkk over all (i, j) pairs."""
    rho_r_t = rho_r(tt).full()  # Convert to NumPy array
    args = [(i, j, tt, dr, Kryk, rho_r_t, nn) for i in range(dr) for j in range(dr)]
    
    with Pool(num_processes) as pool:
        results = pool.map(wsykenkk, args)
    
    return [tt, np.real(sum(results))]



# Main execution via the parallelisation
if __name__ == "__main__":
    data = []
    for tt in time:
        result = parallel_wsykenkk(tt, dr, Kryk, rho_r, nn)
        data.append(result)
        print(result)


    
    
    
    
# Save results
Data = np.array(data)
np.save('/home/ritam.basu/Desktop/hh_coordinate/Db=' + str(db) + 'Dr=' + str(dr) + '.npy', Data) #just copy-paste the path where you want to save the data



#code run-time calculation(can be ignored)
end=tim.time()
print((end-begin)/60)


# Plot results
plt.plot(Data[:, 0], Data[:, 1], '.-', label='db=' + str(db))
plt.legend(loc='upper left', frameon=True, fontsize=8.0)
plt.savefig('/home/ritam.basu/Desktop/hh_coordinate/Db=' + str(db) + 'Dr=' + str(dr) + '.pdf')#just copy-paste the path where you want to save the plot


