def A(xi):
    r"""making $A(t)$"""
    ih0, ih1, h0ci, h1ci, x_k, lindbladian, T, L_I, I = maker(omega_1, H_0, H_1, T_s, Lin, d=2, gamma=0.1):
    
    A = ih0 - h0ci + xi*(ih1 - h1ci) + lindbladian
    return A