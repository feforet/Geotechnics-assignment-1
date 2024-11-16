"""
Calcul de la capacité portante d'un pieu en utilisant la méthode des différences finies
Par Félix Foret, Arnaud Guéguen et Louis Herphelin
Toutes les valeurs sont en unités du système international
"""
import math as m
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

GLOBAL_COUNTER = 0

"""
DATA - OK
"""
### Soil properties
k = 22e6  # N/m^3
gamma_prime = 11e3  # N/m^3
phi_prime = m.radians(35)
alpha = phi_prime / 2
beta = m.radians(45) + phi_prime / 2
K0 = 0.4
Ka = (1 - m.sin(phi_prime)) / (1 + m.sin(phi_prime))
C1 = ((m.tan(beta)**2 * m.tan(alpha)) / m.tan(beta - phi_prime)) + K0 * ((m.tan(phi_prime) * m.sin(beta) / (m.cos(alpha) * m.tan(beta - phi_prime))) + (m.tan(beta) * (m.tan(phi_prime) * m.sin(beta) - m.tan(alpha))))
C2 = m.tan(beta) / m.tan(beta - phi_prime) - Ka
C3 = Ka * ((m.tan(beta)**8 - 1) + K0 * m.tan(phi_prime) * (m.tan(beta)**4))

### Pile properties
E = 210e9
t = lambda D: min (0.00635 + D/100, 0.09)
I = lambda D: m.pi * (D**4 - (D - 2 * t(D))**4) / 64  # a verifier
volume = lambda DL: DL[0] * m.pi * ((DL[1]**2 / 4) - ((DL[1] - 2 * t(DL[1]))**2 / 4))
sigma_max = lambda D: M * (D / 2) / I(D)  # prendre en compte l'effort tranchant ?
f_y = 355e6

### Calculation parameters
# TODO: A ajuster
n_nodes = 10
tolerance = 1e-6
max_iter = 1000
facteur_reduc_y_init = 1e-3

### Optimization parameters
# TODO: A ajuster
DL0 = [10, 100]
DL_history = [np.copy(DL0)]
y_history = []

"""
LOADS - OK
"""
# TODO: A ajuster (même écrire leur formule si besoin genre pour pas arrondir)
M = 0  # Nm
H = 0  # N

"""
p-y curve - OK
"""
def k_yD (y, z, D):
    if y == 0: y = 1e-6
    A = max(3 - 0.8*z/D, 0.9)
    p_u = min((C1*z + C2*D), C3 * D) * gamma_prime * z
    p = A * p_u * np.tanh(k * z * y / (A * p_u))
    return p / y

"""
FINITE DIFFERENCE METHOD - TODO
"""
# E * I * (d^4 y(z) / dz^4) + k_y * D * y(z) = 0

# M(0) = E * I * (d^2 y(0) / dz^2)
# V(0) = E * I * (d^3 y(0) / dz^3)
# M(L) = 0
# V(L) = 0

# f^(4) (x) = (f(x-2h) - 4f(x-h) + 6f(x) - 4f(x+h) + f(x+2h)) / (h^4)

def solve(DL):
    D, L = DL
    h = L / (n_nodes - 1)
    I_D = I(D)
    return tuple([D*L]*n_nodes)  # TODO enlever cette ligne

    z = np.linspace(0, L, n_nodes)
    y = (np.random.rand(n_nodes) - 0.5) / facteur_reduc_y_init
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
    
    ### START - TODO
    ### Le faire dans la bonne rangée de la matrice K
    ### Ajouter M et H dans b
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
    ### END - TODO

    iter = 0
    while True:
        for i in range(2, n_nodes - 2):
            K[i, i] = Kii + k_yD(y[i], z[i], D)

        old_y = y
        y = np.linalg.solve(K, b)

        if (np.linalg.norm(y - old_y) < tolerance):
            break
        if (iter > max_iter):
            print("Maximum number of iterations reached")
            exit()

        iter += 1

    return y

"""
OPTIMIZATION - OK
"""
### utiliser scipy.optimize pour minimiser volume(D, L) sous contrainte de déplacement maximal et de contrainte maximale
def launch_optimization():
    constr = [{'type': 'ineq', 'fun': lambda x: f_y - sigma_max(x[0])}, {'type': 'ineq', 'fun': lambda x: 0.1*x[0] - solve(x)[0]}, {'type': 'ineq', 'fun': lambda x: min(x)}]
    res = sp.optimize.minimize(volume, DL0, constraints=constr, callback=lambda x: DL_history.append(np.copy(x)))
    return res

"""
PLOTTING - OK
"""
### plotter le déplacement y(z) en fonction de z
def plot_displacement(y, z):
    plt.plot(y, -z)
    plt.xlabel('z (m)')
    plt.ylabel('y (m)')
    plt.title('Déplacement du pieu')
    plt.show()

### TODO plotter la contrainte sigma(z) en fonction de z
# Pour ca il faudrait une fonction qui dit le moment en fonction de z

### TODO plotter le moment en fonction de z
# Pour ca il faudrait une fonction qui dit le moment en fonction de z

### TODO plotter l'effort tranchant en fonction de z
# Pour ca il faudrait une fonction qui dit l'effort tranchant en fonction de z

### plotter le volume d'acier en fonction de D
def plot_volume_D(DL_history, vol_hist):
    DL_hist = np.array(DL_history)
    D_hist = DL_hist[:, 0]
    plt.plot(D_hist, vol_hist)
    plt.xlabel('D (m)')
    plt.ylabel('Volume (m^3)')
    plt.title('Volume d\'acier en fonction de D')
    plt.show()

### plotter le volume d'acier en fonction de L
def plot_volume_L(DL_history, vol_hist):
    DL_hist = np.array(DL_history)
    L_hist = DL_hist[:, 1]
    plt.plot(L_hist, vol_hist)
    plt.xlabel('L (m)')
    plt.ylabel('Volume (m^3)')
    plt.title('Volume d\'acier en fonction de L')
    plt.show()

"""
START PROGRAM
"""
result = launch_optimization()
D_opt, L_opt = result.x
print(D_opt, L_opt)
y = solve((D_opt, L_opt))
plot_displacement(y, np.linspace(0, L_opt, n_nodes))
# Peut-etre rajouter D_opt et L_opt dans DL_history (a verifier, mais a priori non)
#DL_history.append([D_opt, L_opt])
vol_hist = np.array([volume(DL) for DL in DL_history])
plot_volume_D(DL_history, vol_hist)
plot_volume_L(DL_history, vol_hist)
for DL in DL_history: y_history.append(solve(DL))