
import numpy as np
import matplotlib.pyplot as plt

d = 5.76 # m 
prof= 20.08 # m
V0 = 2903138.044 + 41814.63 * d + 60994.1552 * d**2
M0 = 520732074.4 + (528676*d + (7.2349*10**5)*d**2)*1.35

Eb = 210 * 10**9 # Pa acier 
Eh = 8648116.172467
t= min(0.00635 + d/100, 0.09)
I = np.pi * (d**4 - (d-2*t)**4)/64   # Moment d'inertie de la poutre cylindrique
lambdaa = (Eh/(4*Eb*I))**0.25

z = np.linspace(0, -prof, 1000)
y = np.exp(-z*lambdaa)*((-V0/(2*(lambdaa**3)*Eb*I) - M0/(2*(lambdaa**2)*Eb*I)) * np.cos(lambdaa*z) + (M0/(2*(lambdaa**2)*Eb*I)) * np.sin(lambdaa*z))

# Plot
plt.plot(y, z)
plt.xlabel('Déplacement latéral (m)')
plt.ylabel('Profondeur (m)')
plt.title('Déplacement latéral du pieu en fonction de la profondeur')
plt.show()
