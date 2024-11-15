"""
Calcul de la capacité portante d'un pieu en utilisant la méthode des différences finies
Par Félix Foret, Arnaud Guéguen et Louis Herphelin
Toutes les valeurs sont en unités du système international (pas encore implémenté):
- longueur: mètres
- force: newtons
- contrainte: pascals
- moment: newton*mètres
- moment d'inertie: mètres^4
- module de Young: pascals
- déplacement: mètres
- angle: radians
- raideur: newtons/mètre
- poids spécifique: newtons/m^3
"""

import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

"""
DATA
"""
### Soil properties
# rate of increase with depth of initial modulus of subgrade reaction
k = 22 * 10**6  # N/m^3
# Submerged soil weight
gamma_prime = 11 * 10**3  # N/m^3
# friction angle
phi_prime = math.radians(35)  # rad
alpha = phi_prime / 2
beta = math.radians(45) + phi_prime / 2
# Coefficients of lateral earth pressure 
K0 = 0.4
Ka = (1 - math.sin(phi_prime)) / (1 + math.sin(phi_prime))

# Pile properties
E = 210 * 10**9  # Pa
t = lambda D: min (0.00635 + D/100, 0.09)
I = lambda D: math.pi * (D**4 - (D - 2 * t(D))**4) / 64
volume = lambda D, L: L * math.pi * ((D**2 / 4) - ((D - 2 * t(D))**2 / 4))

# Calculation parameters
n_nodes = 10
tolerance = 1e-6
max_iter = 1000

# Optimization parameters
# ...

"""
LOADS
"""
M = 0  # Nm
H = 0  # N

"""
API p-y curve
"""
def k_yD (z, y, D):
    C1 = ...
    C2 = ...
    C3 = ...
    A = lambda z: max(3 - 0.8*z/D, 0.9)
    p_u = lambda z: min((C1*z + C2*D) * gamma_prime * z, C3 * D * gamma_prime * z)
    p = lambda y, z: A(z) * p_u(z) * np.tanh(k * z * y / (A(z) * p_u(z)))
    return p(y, z) / y

"""
FINITE DIFFERENCE METHOD
"""
# E * I * (d^4 y(z) / dz^4) + k_y * D * y(z) = 0

# M(0) = E * I * (d^2 y(0) / dz^2)
# V(0) = E * I * (d^3 y(0) / dz^3)
# M(L) = 0
# V(L) = 0

# f^(4) (x) = (f(x-2h) - 4f(x-h) + 6f(x) - 4f(x+h) + f(x+2h)) / (h^4)

def solve(L, D):
    h = L / (n_nodes - 1)
    I_D = I(D)

    # coordonnées des noeuds
    z = np.linspace(0, L, n_nodes)
    y = np.random.rand(n_nodes) / 1000
    b = np.zeros(n_nodes)
    K = np.zeros((n_nodes, n_nodes))
    Kii2 = E * I_D / h**4
    Kii1 = -4 * Kii2
    Kii = 6 * Kii2
    for i in range(2, n_nodes - 2):
        K[i, i-2] = Kii2
        K[i, i-1] = Kii1
        K[i, i+1] = Kii1
        K[i, i+2] = Kii2

    iter = 0
    while iter < max_iter:
        iter += 1
        for i in range(2, n_nodes - 2):
            K[i, i] = Kii + k_yD(z[i], y[i], D)

        ### START - A modifier
        ### A placer avant la boucle while parce que les valeurs ne changent pas
        ### Le faire dans la bonne rangée de la matrice K
        #EI * (2y(0) - 5y(h) + 4y(2h) - y(3h)) / h^2 = M(0)
        K[0, 0] = 2 * E * I_D / h**2
        K[0, 1] = -5 * E * I_D / h**2
        K[0, 2] = 4 * E * I_D / h**2
        K[0, 3] = -E * I_D / h**2
        #EI * (2y(L) - 5y(L-h) + 4y(L-2h) - y(L-3h)) / h^2 = 0
        K[n_nodes-1, n_nodes-1] = 2 * E * I_D / h**2
        K[n_nodes-1, n_nodes-2] = -5 * E * I_D / h**2
        K[n_nodes-1, n_nodes-3] = 4 * E * I_D / h**2
        K[n_nodes-1, n_nodes-4] = -E * I_D / h**2
        #EI * (-y(0) + 3y(h) - 3y(2h) + y(3h)) / h^3 = V(0)
        K[0, 0] += -E * I_D / h**3
        K[0, 1] += 3 * E * I_D / h**3
        K[0, 2] += -3 * E * I_D / h**3
        K[0, 3] += E * I_D / h**3
        #EI * (y(L) - 3y(L-h) + 3y(L-2h) - y(L-3h)) / h^3 = 0
        K[n_nodes-1, n_nodes-1] += E * I_D / h**3
        K[n_nodes-1, n_nodes-2] += -3 * E * I_D / h**3
        K[n_nodes-1, n_nodes-3] += 3 * E * I_D / h**3
        K[n_nodes-1, n_nodes-4] += -E * I_D / h**3
        ### END - A modifier

        new_y = np.linalg.solve(K, b)
        if (np.linalg.norm(new_y - y) < tolerance):
            break
        y = new_y
    return y


"""
OPTIMIZATION
"""
### utiliser scipy.optimize pour minimiser volume(D, L) sous contrainte de déplacement maximal

"""
PLOTTING
"""
### plotter le déplacement y(z) en fonction de z
### plotter la contrainte sigma(z) en fonction de z
### plotter le volume d'acier en fonction de D et L
### plotter le déplacement maximal en fonction de D et L