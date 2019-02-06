
# coding: utf-8

# In[1]:


# # Imports

# ## Qutip imports 1

# In[1]:


from qutip.operators import sigmax, sigmay, sigmaz, identity, qeye
from qutip.operators import position, momentum, num, create, destroy
from qutip.operators import sigmap, sigmam
from qutip.operators import commutator, qdiags
from qutip.tensor import tensor
from qutip.states import basis, ket2dm
from qutip.qip.gates import swap, rx, ry, rz, cnot
'''sqrtnot', 'snot', 'phasegate', 'cphase', 'cnot',
           'csign', 'berkeley', 'swapalpha', 'swap', 'iswap', 'sqrtswap',
           'sqrtiswap', 'fredkin', 'toffoli', 'rotation', 'controlled_gate',
           'globalphase', 'hadamard_transform', 'gate_sequence_product',
           'gate_expand_1toN', 'gate_expand_2toN', 'gate_expand_3toN',
           'qubit_clifford_group']'''
from qutip.qobj import Qobj
from qutip.visualization import hinton
from qutip.visualization import matrix_histogram_complex, matrix_histogram
from qutip.random_objects import rand_herm, rand_unitary, rand_dm
from qutip.operators import jmat , spin_Jy, spin_Jz
from qutip.operators import spin_Jm, spin_Jp, spin_J_set

from qutip.operators import squeeze, squeezing, displace
from qutip.operators import qutrit_ops, qdiags
#from qutip.operators import phase, zero_oper, enr_destroy, enr_identity


# ## Qutip imports 2

# In[3]:


#from 


# ## Numpy imports 

# In[2]:


from numpy import sin, cos, tan, real, imag,  log, conj
from numpy import array, append, linspace, arange
from numpy import add, sqrt, abs, dot
from numpy.random import random, random_sample, rand, seed, RandomState
from numpy import concatenate, hstack, vstack, block, dstack, vsplit
from numpy import trace, diag
from numpy import ones, zeros, ones_like, zeros_like
from numpy import amax, amin, nanmax, nanmin
from numpy import outer, multiply
# from numpy import pi


# ## Scipy imports

# In[17]:


from scipy.integrate import ode, odeint, complex_ode
from scipy.optimize import minimize
from scipy.linalg import eigh, inv, norm
from scipy.linalg import logm, expm
# from scipy.linalg import 
# from scipy import


# ## Matplotlib imports

# In[18]:


from matplotlib.pyplot import plot, figure, show, savefig, axes
from matplotlib.pyplot import xlabel, ylabel, title, legend
from matplotlib import rcParams
from matplotlib.pyplot import style 
from matplotlib.pyplot import xlim, ylim, axis 
# beware not same as axes
from matplotlib.pyplot import subplot, subplots, text
from matplotlib.pyplot import GridSpec
from matplotlib.pyplot import scatter, colorbar


pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
rcParams.update(pgf_with_rc_fonts)
style.use('seaborn-whitegrid')


# ## Math imports

# In[19]:


from math import pi
from math import exp


# ## Cmath imports

# ## Date and datetime imports

# In[20]:


from datetime import date
from datetime import datetime# now


# ## Os imports

# In[21]:


from os import getcwd, mkdir, chdir
from os.path import abspath, join 


# ## Sympy imports

# In[22]:


from sympy import Function, dsolve, Eq, Derivative, symbols
# x, y, z, t = symbols('x y z t')
# k, m, n = symbols('k m n', integer=True)
# f, g, h = symbols('f g h', cls=Function)


# ## Miscellaneous imports

# ## Extra useful functions

# In[23]:


def rint(x):
    print("x = ", x)
    return None


# # Next chapter

# ## sub topic 1

# ## sub topic 2

# ## sub topic 3

# ### sub sub topic 1


