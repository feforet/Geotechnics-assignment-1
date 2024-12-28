"""
Calcul de la capacite portante d'un pieu en utilisant la methode des differences finies
Par Felix Foret, Arnaud Gueguen et Louis Herphelin
Toutes les valeurs sont en unites du systeme international
"""
import math as m
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

"""
Donnees
"""
### Dimensions representees dans l'enonce
h_w = 30  # Profondeur de l'eau
D_h = 6.5  # Diametre de la tour au sommet
D_b = 10  # Diametre de la tour a la base
h_h = 150  # Hauteur de la tour
R = 120  # Longueur des pales
# Diametre de la tour en tout point z
D_tower = lambda z: D_b + (D_b - D_h) * (z + h_w) / h_h

### Proptietes du vent
rho_air = 1.204  # Masse volumique de l'air
C_T = 0.5
C_D = 0.4

### Proprietes des vagues
H_m = 10  # Hauteur significative des vagues
lmbd = 200  # Longueur d'onde
k_w = 2 * m.pi / lmbd  # Nombre d'onde
T = 15  # Periode des vagues
omega = 2 * m.pi / T  # Pulsation des vagues
rho_w = 1030  # Masse volumique de l'eau
C_m = 2
A = lambda D: m.pi * (D**2 / 4)  # Section pleine du pieu

### Proprietes du sol
k = 22e6
gamma_prime = 11e3  # Poids volumique du sol
phi_prime = m.radians(35)  # Angle de frottement interne
alpha = phi_prime / 2
beta = m.radians(45) + phi_prime / 2
K0 = 0.4  # Coefficient de pression au repos
Ka = (1 - m.sin(phi_prime)) / (1 + m.sin(phi_prime))  # Coefficient de pression active
C1 = (((m.tan(beta)**2 * m.tan(alpha)) / m.tan(beta - phi_prime))
      + K0 * ((m.tan(phi_prime) * m.sin(beta) / (m.cos(alpha) * m.tan(beta - phi_prime)))
              + (m.tan(beta) * (m.tan(phi_prime) * m.sin(beta) - m.tan(alpha)))))
C2 = m.tan(beta) / m.tan(beta - phi_prime) - Ka
C3 = Ka * ((m.tan(beta)**8 - 1) + K0 * m.tan(phi_prime) * (m.tan(beta)**4))

### Proprietes du pieu
E = 210e9  # Module de Young de l'acier
t = lambda D: min (0.00635 + D/100, 0.09)  # Epaisseur de la paroi du pieu
I = lambda D: m.pi * (D**4 - (D - 2 * t(D))**4) / 64  # Moment d'inertie du pieu
section = lambda D: m.pi * (D**2 - (D - 2 * t(D))**2) / 4  # Section d'acier du pieu
volume = lambda DL: (DL[1] + h_w) * section(DL[0])  # Volume d'acier du pieu
length = lambda volume, D: volume / section(D) - h_w
f_y = 355e6  # Limite d'elasticite de l'acier

### Parametres de calcul
n_nodes_opti = 200  # Nombre de noeuds
tolerance = 1e-5  # Tolerance de convergence
max_iter = 1e4  # Nombre maximal d'iterations

### Parametres d'optimisation
DL0 = [5.76, 20.08]  # Valeurs initiales de D et L

"""
Charges
"""
# Force sur les pales
U = lambda z: 5 - 0.05 * (z + h_w)
A_r = m.pi * R**2
F_rot = rho_air * A_r * C_T * (U(-h_h-h_w)**2) / 2
bras_lev_rot = (h_h + h_w)
print(f"F_rot = {F_rot:.2f} N \t\tL_rot = {bras_lev_rot:.2f} m \tM_rot = {F_rot*bras_lev_rot:.2f} Nm")

# Force sur la tour
dF_tower = lambda z: rho_air * D_tower(z) * C_D * (U(z)**2) / 2
F_tow = sp.integrate.quad(dF_tower, -h_h-h_w, -h_w)[0]
M_tow = - sp.integrate.quad(lambda z: z * dF_tower(z), -h_h-h_w, -h_w)[0]
print(f"F_tow = {F_tow:.2f} N \t\tL_tow = {M_tow/F_tow:.2f} m \tM_tow = {M_tow:.2f} Nm")

# Force sur le pieu
cos = lambda: 1
sin = lambda: -1
w = lambda z: (H_m / 2) * omega * m.cosh(-k_w*z) / m.sinh(k_w*h_w) * cos()
w_prime = lambda z: -(H_m / 2) * (omega**2) * m.cosh(-k_w*z) / m.sinh(k_w*h_w) * sin()
dF_w = lambda z, D: (rho_w * D * C_D * w(z) * abs(w(z)) / 2) + (C_m * rho_w * A(D) * w_prime(z))
F_w = lambda D: sp.integrate.quad(dF_w, -h_w, 0, args=(D))[0]
M_w = lambda D: - sp.integrate.quad(lambda z: z * dF_w(z, D), -h_w, 0)[0]
print(f"F_w(D={DL0[0]}m) = {F_w(DL0[0]):.2f} N \tL_w(D={DL0[0]}m) = {M_w(DL0[0])/F_w(DL0[0]):.2f} m \tM_w(D={DL0[0]}m) = {M_w(DL0[0]):.2f} Nm")

# Forces totales
H = lambda D: 1.35 * (F_rot + F_tow + F_w(D))  # Effort horizontal en tete du pieu
M = lambda D: 1.35 * ((F_rot * bras_lev_rot) + M_tow + M_w(D))  # Moment en tete du pieu
H_ = lambda D: 2903138.044 + 41814.63*D + 60994.1552*(D**2)
M_ = lambda D: 520732074.4 + 713715.3*D + 977116.5*(D**2)
print(f"H(D={DL0[0]}m) = {H(DL0[0]):.2f} N \tM(D={DL0[0]}m) = {M(DL0[0]):.2f} Nm")

"""
Courbe p-y
"""
def k_y(y, z, D):
    if y == 0: y = 1e-6  # Pour eviter les divisions par 0
    A = max(3 - 0.8*z/D, 0.9)
    p_u = min((C1*z + C2*D), C3 * D) * gamma_prime * z
    p = A * p_u * np.tanh(k * z * y / (A * p_u))
    return p / (y * D)

"""
Methode des differences finies
"""
def solve(DL, n_nodes=n_nodes_opti):
    D, L = DL
    if (D <= 0 or L <= 0): return np.full(n_nodes, np.inf)  # Valeurs invalides
    dz = L / (n_nodes - 1)  # Pas de discretisation
    I_D = I(D)  # Moment d'inertie du pieu

    z = np.linspace(0, L, n_nodes)  # Vecteur des hauteurs
    y = np.ones(n_nodes) / 10  # Vecteur des deplacements
    b = np.zeros(n_nodes)  # Vecteur des forces
    K = np.zeros((n_nodes, n_nodes))  # Matrice de rigidite

    # Remplissage de la matrice de rigidite avec les elements ne dependant pas de y
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
    #EI * (-5y(0) + 18y(h) - 24y(2h) + 14y(3h) - 3y(4h)) / (2h^3) = V(0)
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
    #EI * (5y(L) - 18y(L-h) + 24y(L-2h) - 14y(L-3h) + 3y(L-4h)) / (2h^3) = 0
    K[n_nodes-2, n_nodes-1] = 5 * E * I_D / (2 * dz**3)
    K[n_nodes-2, n_nodes-2] = -18 * E * I_D / (2 * dz**3)
    K[n_nodes-2, n_nodes-3] = 24 * E * I_D / (2 * dz**3)
    K[n_nodes-2, n_nodes-4] = -14 * E * I_D / (2 * dz**3)
    K[n_nodes-2, n_nodes-5] = 3 * E * I_D / (2 * dz**3)

    iter = 0
    while True:
        for i in range(2, n_nodes - 2):
            K[i, i] = Kii + D * k_y(y[i], z[i], D)  # Maj des termes dependants de y

        old_y = y
        y = np.linalg.solve(K, b)

        if (np.linalg.norm(y - old_y) < tolerance):
            break  # Convergence atteinte
        if (np.linalg.norm(y) > 10*D):
            break
        if (iter > max_iter):
            y = np.full(n_nodes, np.inf)
            print(f"Max iterations reached: {D}, {L}")
            break  # Nombre maximal d'iterations atteint
        # Pour eviter les problemes de convergence
        if (iter!=0 and iter % (max_iter//10) == 0):
            print(f"Iteration {iter} reached for D={D}, L={L}")
            y += np.random.rand(n_nodes) * 1e-3
        iter += 1

    return y

"""
Optimisation des dimensions
"""
def launch_optimization():
    starting_d = 4
    best_volume = volume((DL0[0], DL0[1]))
    best_d = DL0[0]
    best_l = DL0[1]
    for i in range((20-4)*100 + 1):
        d = starting_d + i/100
        l_max = length(best_volume, d) * 1.1  # Inutile de chercher si vol > best_volume
        l = optimal_l(d, l_max)
        if (l is None): continue  # l_optimal plus grand que l_max
        vol = volume((d, l))
        if vol < best_volume:
            best_volume = vol
            best_d = d
            best_l = l
            
    DL_opt = (best_d, best_l)
    return DL_opt

def optimal_l(d, l_max):
    l = l_max
    incr = 1
    while incr > 1e-5:
        y = solve((d, l))
        ms, vs, sigmas = M_V_sigma(y, np.linspace(0, l, n_nodes_opti), d)
        if (y[0] > 0.1*d) or (max(sigmas) > f_y):
            l += incr
            incr /= 10
        l -= incr
        if l > l_max: return None
    return l

"""
Plots
"""
### calcul des mmoments et efforts tranchants
def M_V_sigma(y, z, D):
    dydz = np.gradient(y, z, edge_order=2)
    d2ydz2 = np.gradient(dydz, z, edge_order=2)
    Ms = E * I(D) * d2ydz2
    Ms[0] = M(D)  # Moment en tete
    Ms[-1] = 0  # Moment en pied
    Vs = np.gradient(Ms, z, edge_order=2)
    Vs[0] = H(D)  # Effort en tete
    Vs[-1] = 0  # Effort en pied
    sigmas_bending = Ms * (D / 2) / I(D)
    sigmas_shear = Vs / section(D)
    sigmas = sigmas_bending/2 + np.sqrt((sigmas_bending/2)**2 + sigmas_shear**2)
    return Ms, Vs, sigmas

### plotter le deplacement y(z) en fonction de z
def plot_displacement(ys, zs):
    if (np.copy(ys[0]).ndim == 0): 
        ys = [ys]
        zs = [zs]
    for i, y in enumerate(ys):
        plt.plot(y*1e2, -zs[i], label = f'{len(y)} noeuds')
    plt.xlabel('Déplacement latéral [cm]')
    plt.ylabel('Profondeur (-z) [m]')
    plt.title('Déplacement du pieu')
    plt.legend()
    plt.savefig('plots/displacement_'+str(time.strftime("%Y_%m_%d_%H_%M_%S"))+'.png')
    plt.show()

### plotter la contrainte sigma(z) en fonction de z
def plot_stresses(sigmas, z):
    plt.plot(sigmas/1e6, -z)
    plt.xlabel('Contrainte [MPa]')
    plt.ylabel('Profondeur (-z) [m]')
    plt.title('Contrainte du pieu')
    plt.show()

### plotter le moment en fonction de z
def plot_moments(Ms, z):
    plt.plot(Ms/1e6, -z)
    plt.xlabel('Moment [MNm]')
    plt.ylabel('Profondeur (-z) [m]')
    plt.title('Moment flechissant du pieu')
    plt.show()

### plotter l'effort tranchant en fonction de z
def plot_shear_forces(Vs, z):
    plt.plot(Vs/1e6, -z)
    plt.xlabel('Effort tranchant [MN]')
    plt.ylabel('Profondeur (-z) [m]')
    plt.title('Effort tranchant du pieu')
    plt.show()

### plotter la courbe p-y pour z et D donnes
def plot_ky(z, D):
    plt.plot(np.linspace(0, 1, 100), [k_y(y, z, D) for y in np.linspace(0, 1, 100)])
    plt.xlabel('y')
    plt.ylabel('k_y [N/m^3]')
    plt.title(f'Courbe p-y pour z={z}m et D={D}m')
    plt.show()
    plt.plot(np.linspace(0, 1, 100), [D*y*k_y(y, z, D) for y in np.linspace(0, 1, 100)])
    plt.xlabel('y')
    plt.ylabel('p [N/m]')
    plt.title(f'Courbe p-y pour z={z}m et D={D}m')
    plt.show()

"""
Demarrage du programme
"""
# p-y curve
plot_ky(10, 5.76)

# n_nodes optimization
n_nodes = [6, 10, 25, 50, 100, 200, 1000]
ys = []
zs = []
times = []
for i, nnodes in enumerate(n_nodes):
    start = time.perf_counter_ns()
    y = solve(DL0, nnodes)
    end = time.perf_counter_ns()
    z = np.linspace(0, DL0[1], nnodes)
    ys.append(y)
    zs.append(z)
    times.append(end - start)

print("\n" + "-"*80 + "\n")
print(f"y({n_nodes[-1]}): {ys[-1][0]}")
print(f"y({n_nodes[-2]}): {ys[-2][0]}")
print(f"temps de calcul pour {n_nodes[-1]} noeuds: {times[-1]/1e9} s")
print(f"temps de calcul pour {n_nodes[-2]} noeuds: {times[-2]/1e9} s")
print("\n" + "-"*80 + "\n")

plot_displacement(ys, zs)

# Optimal dimensions
y = solve(DL0)
y_tete = y[0]
z = np.linspace(0, DL0[1], n_nodes_opti)
Ms, Vs, sigmas = M_V_sigma(y, z, DL0[0])
sigma_max = max(sigmas)
z_sigma_max = z[np.argmax(sigmas)]  # Profondeur de la contrainte maximale
print(f"Volume(D={DL0[0]}m, L={DL0[1]}m) = {volume(DL0):.2f} m^3")
print(f"sigma_max(D={DL0[0]}m, z={z_sigma_max:.4f}m) = {sigma_max:.2f} N/m^2")
print(f"deplacement en tete = {y_tete:.6f} m")
plot_displacement(y, z)
plot_moments(Ms, z)
plot_shear_forces(Vs, z)
plot_stresses(sigmas, z)

# Dimensions optimization
DL_opt = launch_optimization()
D_opt, L_opt = DL_opt
y_opt = solve((D_opt, L_opt))
y_tete_opt = y[0]
z_opt = np.linspace(0, L_opt, n_nodes_opti)
Ms_opt, Vs_opt, sigmas_opt = M_V_sigma(y_opt, z_opt, D_opt)
sigma_max_opt = max(sigmas_opt)
z_sigma_max_opt = z_opt[np.argmax(sigmas_opt)]  # Profondeur de la contrainte maximale

print("\n" + "-"*80 + "\nOptimization result:")
print(f"D = {D_opt:.4f} m \tL = {L_opt:.4f} m")
print(f"Volume(D={D_opt:.4f}m, L={L_opt:.4f}m) = {volume((D_opt, L_opt)):.4f} m^3")
print(f"sigma_max(D={D_opt:.4f}m, z={z_sigma_max_opt:.4f}m) = {sigma_max_opt:.4f} N/m^2")
print(f"Deplacement en tete = {y_tete_opt:.6f} m")
print(f"y[0] = {y_tete_opt/D_opt:.6f} * D")
plot_displacement(y, np.linspace(0, L_opt, n_nodes_opti))
plot_moments(Ms_opt, z_opt)
plot_shear_forces(Vs_opt, z_opt)
plot_stresses(sigmas_opt, z_opt)
