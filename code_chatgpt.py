import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Constantes
rho_air = 1.225  # Masse volumique de l'air en kg/m^3
rho_water = 1030  # Masse volumique de l'eau en kg/m^3
rotor_radius = 120  # Rayon du rotor en m (approximé pour une éolienne de 15MW)
A_R = np.pi * (rotor_radius**2)  # Aire balayée par le rotor en m^2
C_T = 0.5  # Coefficient de poussée du rotor
C_D_tow = 0.4  # Coefficient de traînée de la tour
U = 15  # Vitesse du vent en m/s
H_m = 10  # Hauteur des vagues en m
lambda_wave = 200  # Longueur d'onde en m
T = 15  # Période des vagues en s
omega = 2 * np.pi / T  # Fréquence angulaire des vagues en rad/s
hw = 30  # Profondeur de l'eau en m
D_tow = 6  # Diamètre approximé de la tour en m

# Calcul de la force de poussée du rotor
F_rotor = 0.5 * rho_air * A_R * C_T * U**2
print("Force de poussée du rotor (N):", F_rotor)

# Calcul de la force de traînée de la tour (par mètre de hauteur)
F_tower_drag = 0.5 * rho_air * D_tow * C_D_tow * U**2
print("Force de traînée de la tour par mètre (N/m):", F_tower_drag)

# Calcul de la force due aux vagues (force de Morison)
def wave_force(D, z):
    k = 2 * np.pi / lambda_wave
    w = H_m * omega / 2 * (np.cosh(k * (hw - z)) / np.sinh(k * hw))
    F_wave_drag = 0.5 * rho_water * D * C_D_tow * w**2
    F_wave_inertia = 2 * rho_water * (np.pi * D**2 / 4) * (omega**2 * H_m / (2 * T))
    return F_wave_drag + F_wave_inertia

# Intégration pour la force totale des vagues sur la hauteur d'eau
z_values = np.linspace(0, hw, 100)  # 100 points de profondeur
wave_forces = [wave_force(D_tow, z) for z in z_values]
H_wave_total = np.trapz(wave_forces, z_values)  # Intégration numérique

# Somme des forces horizontales
H_total = F_rotor + H_wave_total
print("Force horizontale totale (N):", H_total)


def p_y_curve(y, z, D):
    A = max(3 - 0.8 * (z / D), 0.9)
    k = 22e6  # Module de réaction de sol initial pour le sable en MN/m^3
    phi_prime = 30  # Angle de frottement du sable
    gamma_prime = 11  # Poids volumique du sol immergé en kN/m^3
    
    # Calcul des coefficients API
    C1, C2, C3 = calculate_C_coefficients(phi_prime)
    pu = min((C1 * z + C2 * D) * gamma_prime * z, C3 * D * gamma_prime * z)
    
    p = A * pu * np.tanh(k * z / (A * pu) * y)
    return p / y  # secant stiffness Dky


def calculate_C_coefficients(phi_prime):
    # Convertir l'angle de frottement en radians
    phi_prime_rad = np.radians(phi_prime)
    
    # Calculer les valeurs des angles α et β
    alpha = phi_prime_rad / 2
    beta = np.radians(45 + phi_prime / 2)
    
    # Coefficients de pression latérale
    K0 = 0.4
    Ka = (1 - np.sin(phi_prime_rad)) / (1 + np.sin(phi_prime_rad))
    
    # Calcul des coefficients C1, C2 et C3
    C1 = ((np.tan(beta)**2 * np.tan(alpha)) / np.tan(beta - phi_prime_rad) 
          + K0 * ((np.tan(phi_prime_rad) * np.sin(beta)) / (np.cos(alpha) * np.tan(beta - phi_prime_rad)) 
          + np.tan(beta) * (np.tan(phi_prime_rad) * np.sin(beta) - np.tan(alpha))))
    
    C2 = np.tan(beta) / np.tan(beta - phi_prime_rad) - Ka
    C3 = Ka * ((np.tan(beta)**8 - 1) + K0 * np.tan(phi_prime_rad) * (np.tan(beta)**4))
    
    return C1, C2, C3

def finite_difference_matrix(N, EI, ky_values):
    dz = L / N  # Discrétisation
    K = np.zeros((N, N))
    for i in range(1, N - 1):
        K[i, i - 1] = EI / dz**4
        K[i, i] = -2 * EI / dz**4 + ky_values[i] * D
        K[i, i + 1] = EI / dz**4
    
    # Conditions aux limites
    K[0, 0] = EI / dz**4  # Moment nul
    K[-1, -1] = EI / dz**4  # Cisaillement nul
    return K

# Paramètres
N = 50  # Nombre de nœuds
L = 30  # Longueur du monopieu en m
D = 6  # Diamètre du monopieu en m

# Génération des valeurs de ky en fonction de la profondeur z
ky_values = [p_y_curve(y, z, D) for z in np.linspace(0, L, N)]

# Matrice de rigidité
K = finite_difference_matrix(N, EI, ky_values)




def objective(params):
    D, L = params
    # Calcul de la masse ou du volume d'acier
    steel_volume = np.pi * D * L * thickness
    return steel_volume

constraints = [{'type': 'ineq', 'fun': lambda x: 0.1 * x[0] - displacement}]  # Contrainte de déplacement

# Optimisation
result = minimize(objective, x0=[6, 30], constraints=constraints)
optimal_D, optimal_L = result.x




# Graphique de la forme déformée
plt.plot(displacement_profile, depths)
plt.xlabel("Déplacement (m)")
plt.ylabel("Profondeur (m)")
plt.title("Forme déformée du monopieu optimisé")
plt.show()

# Graphiques de la force de cisaillement et du moment de flexion
plt.plot(shear_force, depths, label="Cisaillement")
plt.plot(bending_moment, depths, label="Moment de flexion")
plt.xlabel("Force (kN) / Moment (kN·m)")
plt.ylabel("Profondeur (m)")
plt.legend()
plt.show()
