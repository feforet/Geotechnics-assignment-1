"""
Calcul de la capacité portante d'un pieu en utilisant la méthode des différences finies
Par Félix Foret, Arnaud Guéguen et Louis Herphelin
Toutes les valeurs sont en unités du système international
"""
import math as m
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

"""
Données
"""
### Dimensions représentées dans l'enonce
h_w = 30  # Profondeur de l'eau
D_h = 6.5  # Diametre de la tour au sommet
D_b = 10  # Diametre de la tour a la base
h_h = 150  # Hauteur de la tour
R = 120  # Longueur des pales
D_tower = lambda z: D_b + (D_b - D_h) * (z + h_w) / h_h  # Diamètre de la tour en tout point z

### Proptiétés du vent
rho_air = 1.204  # Masse volumique de l'air
C_T = 0.5
C_D = 0.4

### Propriétés des vagues
H_m = 10  # Hauteur significative des vagues
lmbd = 200  # Longueur d'onde
k_w = 2 * m.pi / lmbd  # Nombre d'onde
T = 15  # Période des vagues
omega = 2 * m.pi / T  # Pulsation des vagues
rho_w = 1030  # Masse volumique de l'eau
C_m = 2
A = lambda D: m.pi * (D**2 / 4)  # Section pleine du pieu

### Propriétés du sol
k = 22e6
gamma_prime = 11e3  # Poids volumique du sol
phi_prime = m.radians(35)  # Angle de frottement interne
alpha = phi_prime / 2
beta = m.radians(45) + phi_prime / 2
K0 = 0.4  # Coefficient de pression au repos
Ka = (1 - m.sin(phi_prime)) / (1 + m.sin(phi_prime))  # Coefficient de pression active
C1 = ((m.tan(beta)**2 * m.tan(alpha)) / m.tan(beta - phi_prime)) + K0 * ((m.tan(phi_prime) * m.sin(beta) / (m.cos(alpha) * m.tan(beta - phi_prime))) + (m.tan(beta) * (m.tan(phi_prime) * m.sin(beta) - m.tan(alpha))))
C2 = m.tan(beta) / m.tan(beta - phi_prime) - Ka
C3 = Ka * ((m.tan(beta)**8 - 1) + K0 * m.tan(phi_prime) * (m.tan(beta)**4))

### Propriétés du pieu
E = 210e9  # Module de Young de l'acier
t = lambda D: min (0.00635 + D/100, 0.09)  # Epaisseur de la paroi du pieu
I = lambda D: m.pi * (D**4 - (D - 2 * t(D))**4) / 64  # Moment d'inertie du pieu
section = lambda D: m.pi * (D**2 - (D - 2 * t(D))**2) / 4  # Section d'acier du pieu
volume = lambda DL: DL[1] * m.pi * section(DL[0])  # Volume d'acier du pieu
sigma_max_bending = lambda D: M(D) * (D / 2) / I(D)  # Contrainte de flexion maximale en z=0
sigma_max_shear = lambda D: H(D) / section(D)  # Contrainte de cisaillement maximale en z=0
sigma_max = lambda D: (sigma_max_bending(D) / 2) + m.sqrt((sigma_max_bending(D) / 2)**2 + sigma_max_shear(D)**2)  # Contrainte maximale en z=0
f_y = 355e6  # Limite d'élasticité de l'acier

### Paramètres de calcul
n_nodes = 10**3  # Nombre de noeuds
tolerance = 1e-4  # Tolérance de convergence
max_iter = 1e3  # Nombre maximal d'itérations

### Paramètres d'optimisation
DL0 = [6, 35]  # Valeurs initiales de D et L

"""
Charges
"""
# Force sur les pales
U = lambda z: 5 - 0.05 * (z + h_w)
A_r = m.pi * R**2
F_rot = rho_air * A_r * C_T * (U(-h_h-h_w)**2) / 2
bras_lev_rot = (h_h + h_w)
print(f"F_rot = {F_rot:.2f} N \t\tL_rot = {bras_lev_rot:.2f} m \tM_rot = {F_rot*bras_lev_rot:.2f} Nm")  # Pour vérifier

# Force sur la tour
dF_tower = lambda z: rho_air * D_tower(z) * C_D * (U(z)**2) / 2
F_tow = sp.integrate.quad(dF_tower, -h_h-h_w, -h_w)[0]
M_tow = - sp.integrate.quad(lambda z: z * dF_tower(z), -h_h-h_w, -h_w)[0]
print(f"F_tow = {F_tow:.2f} N \t\tL_tow = {M_tow/F_tow:.2f} m \tM_tow = {M_tow:.2f} Nm")  # Pour vérifier

# Force sur le pieu
cos = lambda: 1
sin = lambda: -1
w = lambda z: (H_m / 2) * omega * m.cosh(-k_w*z) / m.sinh(k_w*h_w) * cos()
w_prime = lambda z: -(H_m / 2) * (omega**2) * m.cosh(-k_w*z) / m.sinh(k_w*h_w) * sin()
dF_w = lambda z, D: (rho_w * D * C_D * w(z) * abs(w(z)) / 2) + (C_m * rho_w * A(D) * w_prime(z))
F_w = lambda D: sp.integrate.quad(dF_w, -h_w, 0, args=(D))[0]
M_w = lambda D: - sp.integrate.quad(lambda z: z * dF_w(z, D), -h_w, 0)[0]
print(f"F_w(D={DL0[0]}m) = {F_w(DL0[0]):.2f} N \tL_w(D={DL0[0]}m) = {M_w(DL0[0])/F_w(DL0[0]):.2f} m \tM_w(D={DL0[0]}m) = {M_w(DL0[0]):.2f} Nm")  # Pour vérifier

# Forces totales
H_ = lambda D: 1.35 * (F_rot + F_tow + F_w(D))
M_ = lambda D: 1.35 * ((F_rot * bras_lev_rot) + M_tow + M_w(D))
H = lambda D: 2903138.044 + 41814.63*D + 60994.1552*(D**2)  # Effort horizontal en tete du pieu
M = lambda D: 520732074.4 + (528678*D + 723490*(D**2)) * 1.35  # Moment en tete du pieu
print(f"H(D={DL0[0]}m) = {H(DL0[0]):.2f} N \tM(D={DL0[0]}m) = {M(DL0[0]):.2f} Nm")

"""
Courbe p-y
"""
def k_yD (y, z, D):
    if y == 0: y = 1e-6  # Pour éviter les divisions par 0
    A = max(3 - 0.8*z/D, 0.9)
    p_u = min((C1*z + C2*D), C3 * D) * gamma_prime * z
    p = A * p_u * np.tanh(k * z * y / (A * p_u))
    return p / y

"""
Methode des differences finies
"""
def solve(DL, verbose=True):
    if verbose: print(f"Solving for D={DL[0]} and L={DL[1]}")
    D, L = DL
    if (D <= 0 or L <= 0): return np.full(n_nodes, np.inf)  # Si D ou L sont négatifs ou nuls, on retourne un vecteur infini
    dz = L / (n_nodes - 1)  # Pas de discrétisation
    I_D = I(D)  # Moment d'inertie du pieu

    z = np.linspace(0, L, n_nodes)  # Vecteur des hauteurs
    y = np.ones(n_nodes) / 10  # Vecteur des déplacements
    b = np.zeros(n_nodes)  # Vecteur des forces
    K = np.zeros((n_nodes, n_nodes))  # Matrice de rigidité

    # Remplissage de la matrice de rigidité
    Kii2 = E * I_D / dz**4
    Kii1 = -4 * Kii2
    Kii = 6 * Kii2
    for i in range(2, n_nodes - 2):
        K[i, i-2] = Kii2
        K[i, i-1] = Kii1
        K[i, i+1] = Kii1
        K[i, i+2] = Kii2
    
    # Conditions aux limites
    #EI * (2y(0) - 5y(h) + 4y(2h) - y(3h)) / h^2 = M(0)
    K[0, 0] = 2 * E * I_D / dz**2
    K[0, 1] = -5 * E * I_D / dz**2
    K[0, 2] = 4 * E * I_D / dz**2
    K[0, 3] = -E * I_D / dz**2
    b[0] = M(D)
    #EI * (-y(0) + 3y(h) - 3y(2h) + y(3h)) / h^3 = V(0)
    K[1, 0] = -5 * E * I_D / (2 * dz**3)
    K[1, 1] = 18 * E * I_D / (2 * dz**3)
    K[1, 2] = -24 * E * I_D / (2 * dz**3)
    K[1, 3] = 14 * E * I_D / (2 * dz**3)
    K[1, 4] = -3 * E * I_D / (2 * dz**3)
    b[1] = H(D)
    #EI * (2y(L) - 5y(L-h) + 4y(L-2h) - y(L-3h)) / h^2 = 0
    K[n_nodes-1, n_nodes-1] = 2 * E * I_D / dz**2
    K[n_nodes-1, n_nodes-2] = -5 * E * I_D / dz**2
    K[n_nodes-1, n_nodes-3] = 4 * E * I_D / dz**2
    K[n_nodes-1, n_nodes-4] = -E * I_D / dz**2
    #EI * (y(L) - 3y(L-h) + 3y(L-2h) - y(L-3h)) / h^3 = 0
    K[n_nodes-2, n_nodes-1] = 5 * E * I_D / (2 * dz**3)
    K[n_nodes-2, n_nodes-2] = -18 * E * I_D / (2 * dz**3)
    K[n_nodes-2, n_nodes-3] = 24 * E * I_D / (2 * dz**3)
    K[n_nodes-2, n_nodes-4] = -14 * E * I_D / (2 * dz**3)
    K[n_nodes-2, n_nodes-5] = 3 * E * I_D / (2 * dz**3)

    iter = 0
    while True:
        if verbose: print(iter)
        for i in range(2, n_nodes - 2):
            K[i, i] = Kii + k_yD(y[i], z[i], D)  # Maj des termes dependants de y

        old_y = y
        y = np.linalg.solve(K, b)

        if (np.linalg.norm(y - old_y) < tolerance):
            break  # Convergence atteinte
        if (np.linalg.norm(y) > D): 
            break  # Déplacement largement supérieur à la limite
        if (iter > max_iter):
            y = np.full(n_nodes, np.inf)
            break  # Nombre maximal d'itérations atteint
        iter += 1

    return y

"""
Optimisation
"""
def launch_optimization():  # Problèmes de convergence
    constr = [{'type': 'ineq', 'fun': lambda x: f_y - sigma_max(x[0])},  # Contrainte max
              {'type': 'ineq', 'fun': lambda x: 0.1*x[0] - abs(solve(x)[0])}]  # Déplacement en tete max
    bounds = [(4, None), (10, None)]
    res = sp.optimize.minimize(volume, DL0, constraints=constr, bounds=bounds)
    print(res)
    return res.x
def my_optimization(dmin=0, dmax=30, ds=31, lmin=0, lmax=100, ls=101):
    DL_y_sigma_V_Accept = [{'DL': (d, l), 'V': 0, 'Acceptable': False}for l in np.linspace(lmin, lmax, ls) for d in np.linspace(dmin, dmax, ds)]
    for dict in DL_y_sigma_V_Accept:
        D, L = dict['DL']
        dict['V'] = volume((D,L))  # Volume d'acier
        y_tete = solve((D,L), verbose=False)[0]  # Deplacement en tete
        sigma = sigma_max(D)  # Contrainte max
        dict['Acceptable'] = f_y - sigma > 0 and 0.1*D - abs(y_tete) > 0
    # Suppression des combinaisons non acceptables
    filtered_dict = np.array([dict for dict in DL_y_sigma_V_Accept if dict['Acceptable']])
    # Tri par ordre croissant de volume
    sorted_dict = sorted(filtered_dict, key=lambda x: x['V'])

    return sorted_dict

"""
Plots
"""
### plotter le déplacement y(z) en fonction de z
def plot_displacement(y, z):
    plt.plot(y, -z)
    plt.xlabel('Déplacement latéral [m]')
    plt.ylabel('Profondeur (-z) [m]')
    plt.title('Déplacement du pieu')
    plt.show()

### TODO plotter la contrainte sigma(z) en fonction de z
# Pour ca il faudrait une fonction qui dit le moment en fonction de z

### TODO plotter le moment en fonction de z
# Pour ca il faudrait une fonction qui dit le moment en fonction de z

### TODO plotter l'effort tranchant en fonction de z
# Pour ca il faudrait une fonction qui dit l'effort tranchant en fonction de z

"""
Démarrage du programme
"""
y = solve(DL0)
print(f"Volume(D={DL0[0]}m, L={DL0[1]}m) = {volume(DL0):.2f} m^3")
print(f"sigma_max(D={DL0[0]}m) = {sigma_max(DL0[0]):.2f} N/m^2")
print(f"deplacement en tete = {y[0]:.6f} m")
plot_displacement(y, np.linspace(0, DL0[1], n_nodes))


dmin, dmax, ds = 3, 10, 8
lmin, lmax, ls = 10, 30, 21
DL_y_sigma_V_Accept = my_optimization(dmin=dmin, dmax=dmax, ds=ds, lmin=lmin, lmax=lmax, ls=ls)
D_opt, L_opt = DL_y_sigma_V_Accept[0]['DL']
print("-"*80 + "\nOptimization result:")
print(f"D_min={dmin}m, D_max={dmax}m, D_step={ds}")
print(f"L_min={lmin}m, L_max={lmax}m, L_step={ls}")
print(f"D = {D_opt:.2f} m \tL = {L_opt:.2f} m")
print(f"Volume(D={D_opt:.2f}m, L={L_opt:.2f}m) = {volume((D_opt, L_opt)):.2f} m^3")
print(f"sigma_max(D={D_opt}) = {sigma_max(D_opt):.2f} N/m^2")
y = solve((D_opt, L_opt), verbose=False)
print(f"Deplacement en tete = {y[0]:.6f} m")
print(f"y[0] = {y[0]/D_opt:.6f} * D")
plot_displacement(y, np.linspace(0, L_opt, n_nodes))
plt.show()