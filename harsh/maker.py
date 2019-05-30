# File to emulate maker
# import jate.py
from qutip.operators import sigmax, sigmay, sigmaz, identity, qeye
from qutip.operators import position, momentum, num, create, destroy
from numpy import array, append, linspace, arange
from numpy import sin, cos, tan, real, imag,  log, conj
from math import pi
# from qutip.operators import sigmap, sigmam
# from qutip.operators import commutator, qdiags
from qutip.tensor import tensor

from numpy import sin, cos, tan, real, imag,  log, conj
from numpy import array, append, linspace, arange
from numpy import add, sqrt, abs, dot
from numpy.random import random, random_sample, rand, seed, RandomState
from numpy import concatenate, hstack, vstack, block, dstack, vsplit
from numpy import trace, diag
from numpy import ones, zeros, ones_like, zeros_like
from numpy import amax, amin, nanmax, nanmin
from numpy import outer, multiply
from numpy import convolve, clip
from numpy import vectorize
   
def maker(omega_1, H_0, H_1, T_s, Lin, d=2, gamma=0.1):
    r"""maker
    Makes all the things that remain constant throught the program, but are 
    repeatedly used.
    

    Parameters
    ----------
    omega_1 : float
              frequency corresponding to half of the difference between 
              energy levels of the qubit
              
    H_0     : Qobj
              Bare Hamiltonian 
              
    H_1     : Qobj
              Interaction Hamiltonian 
              
    T_s     : Qobj
              Unitary to be implemented in the Hilbert space
    
    Lin     : Qobj
              Linbladian operators

    d       : int
              Dimension of the matrix. Defaults to 2
    
    gamma   : float
              Damping constant of the Linbladian

    
    Returns
    -------
    
    ih0     : Qobj
              $I\otimes H_{0}$
              
    ih1     : Qobj
              $I\otimes H_{1}$

    h0ci    : Qobj
              $H_{0}^{*}\otimes I $

    h1ci    : Qobj
              $H_{1}^{*}\otimes I $

    T       : Qobj
              Target unitary transformed to the Liouville space

    linbladian : Qobj
                 The full lindbladian term as it appears on transformation to 
                 the Liouville space.
        
    """
    I = identity(d)
    L_I = tensor(I, I)
    ih0 = tensor(I, H_0) 
    ih1 = tensor(I, H_1) 
    h0ci = tensor(H_0.conj(), I) 
    h1ci = tensor(H_1.conj(), I)
    term1 = tensor(Lin.trans(), Lin)
    # print("Calculating term1", term1)
    term2 = tensor(I, ((Lin.dag())*(Lin)))
    # print("Calculating term2", term2)
    term3 = tensor(((Lin.trans())*(Lin.conj())), I)
    # print("Calculating term3", term3)
    lindbladian = 1j*(gamma)*(term1 - 0.5*(term2 + term3))
    # print("Calculating lindbladian", lindbladian)    
    x_k = ih1 - h1ci
    # B\rho C --> C*\otimes B
    # T \rho T.dag -->  T.trans \otimes T
    T = tensor(T_s.trans(), T_s) # Transforming $T_{s}$ to liouville space
    # print("Calculating T", T)        
    
    return ih0, ih1, h0ci, h1ci, x_k, lindbladian, T, L_I, I
    
# maker(omega_1, H_0, H_1, T_s, Lin, d=2, gamma=0.1):
# T_s1 = 0.1*sigmax() + 0.5*sigmaz()
# Lin1 = 0.2*sigmax() + 0.7*sigmaz()
 


def A(xi):
    """making $A(t)$"""
    A = ih0 - h0ci + xi*(ih1 - h1ci) + lindbladian
    return A

def L(xi, dt):
    r"""Making $L(t) from $A(t)$"""
    L = (-1j*A(xi)*dt).expm()
    return L

# building the function to optimize (optimizee)
def L_vec(xi_vec, dt):
    r"""Building the vector of differential $L(t)$"""
    L_vec = [L(xi, dt) for xi in xi_vec] 
    return L_vec

def fidelity_calc(A, B):
    r"""Making a generalised fidelity function"""
    # Formula on paper: F[g] = −Tr((A − B)†(A − B))
    f = ((A - B).dag() * (A - B)).tr() # Tejas Shetty
    # f = (A.dag() * B).tr()/A.shape[0] # By Paul D. Nation
    return f

def L_full_maker(xi_vec, dt):
    r"""Building the $L(t)$ for the total time $t$"""
    xi_vec_size = xi_vec.size # finding the size of xi
    L_full = L_I # Identity for the for loop of L
    L_v = L_vec(xi_vec, dt) # calling L_vec
    for i in range(xi_vec_size): # generating L_full
        L_full = L_full*L_v[xi_vec_size - 1 - i]
    return L_full

def F(xi_vec, dt):
    r"""Using the fidelity metric to find out the closeness between $T$
    and $L(t)$"""
    L_full = L_full_maker(xi_vec, dt)
    F = real(-fidelity_calc(T, L_full))   
    return F

def L_comma_k_maker(xi_vec, k, dt):
    r"""Making of the derivative of full $L(t)$ at time $t_{k}$
        Ldk = LNLN−1 ...XkLk ...L2L1
    """
    N = xi_vec.size 
    if(k<1 or k>N):
        raise Exception("k should be between 1 to {}".format(N))
    # Determining the size of xi, and thus the time_steps indirectly.
    L_v = L_vec(xi_vec, dt)# Making of the full $L(t)$
    inner_part = L_I # Beginner for the for loop
    #print("i\tN\tk\tinner_part")
    for i in range(N):
        ptr = N - 1 - i;
        #print("Current Pointer ",ptr)
        if ptr == (k - 1):
            # The step at which $X_{k}(t)$ has to be inserted 
            #print("Matching with k Pointer ",ptr)
            inner_part = inner_part*x_k*L_v[ptr]
            #print(i,"\t",N,"\t",k,"\t",inner_part)
        else:
            # Usual multiplications of $L_{k}$
            inner_part = inner_part*L_v[ptr]
            #print(i,"\t",N,"\t",k,"\t",inner_part)
    l_comma_k = inner_part
    return l_comma_k

def updater(xi_vec, dt, epsilon):
    r"""Implementing the GRAPE update step"""
    # print('IN xi_vec = ', xi_vec)
    xi_vec_size = xi_vec.size # finding the size of xi
    L_full = L_full_maker(xi_vec, dt)
    di = []
    for k in range(xi_vec_size):
        # Building the thing to be added to the old function
        L_comma_k = L_comma_k_maker(xi_vec, (k+1), dt)
        differentiated = T - L_comma_k
        plain = T - L_full
        c = -differentiated.dag()*plain
        d = -plain.dag()*differentiated
        inside = c.tr() + d.tr()
        di.append(epsilon*inside)

    diff = array(di)
    xi_new_vec = xi_vec + diff
    # print('OUT xi_new_vec = ', xi_new_vec)
    return diff, xi_new_vec 

def terminator(max_iter, time_steps, total_time, epsilon=2*pi*0.1):
    r"""Brief description of the function"""
    
    #print('terminator inputs: ', 'max_iter=',max_iter, 'time_steps=',time_steps, 'total_time=',total_time, 'epsilon=',epsilon)
    # 1000*random_sample((time_steps,))
    
    xi_initial = array([0.53694653,0.8132245,0.08551491,0.60347567,0.94720689,0.41482569,0.94908062])
    #xi_initial = random_sample((time_steps,)

    #total_time/time_steps
    dt = total_time/ xi_initial.size
    #print('dt = ', dt)
    xi_diff, xi_new_vec = updater(xi_initial, dt, epsilon)
    min_iter = int(max_iter/2)
    f = F(xi_new_vec, dt)
    #print('First Fidility=',f)
    
    for i in range(max_iter):
        #print('idx', i)
        xi_diff, xi_new_vec = updater(xi_new_vec, dt, epsilon)
        f = F(xi_new_vec, dt)
        print('Updated Fidility=',f)
        #print('xi_new_vec=', xi_new_vec)
        #print('xi_diff=', xi_diff)
        #print('epsilon=', epsilon)
        #print(amax(xi_diff))

    return xi_new_vec

def tester():
    # Testing major functions 
    # xi_vec_test = array([1.0, 2.0]) 
    # l_comma_k = L_comma_k_maker(xi_vec_test, 2, 0.001)
    # diff, xi_new_vec = updater(xi_vec_test, 0.001, 0.001)
    total_time_evo = 2*pi # total time allowed for evolution
    times = linspace(0, total_time_evo, 7)
    epsilon = 0.001

    xi_opt = terminator(10, len(times), total_time_evo,epsilon)
    #print('xi_opt=', xi_opt)

## Setting up base data
omega_1 = 0.5
H_0 = omega_1*sigmaz() 
H_1 = sigmay()
T_s = sigmax() 
Lin = sigmaz()
ih0, ih1, h0ci, h1ci, x_k, lindbladian, T, L_I, I = maker(omega_1, H_0, H_1, T_s, Lin, d=2, gamma=0.1)

#print( 'ih0=', ih0, 'ih1=', ih1, 'h0ci=', h0ci, 'h1ci=', h1ci, 'x_k=', x_k, 'lindbladian=', lindbladian, 'T=', T, 'L_I=', L_I, 'I',I)

tester()