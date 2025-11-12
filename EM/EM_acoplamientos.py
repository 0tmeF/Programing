import numpy as np

# Parámetros (usar float explícito)
P_max = 9.0 # Watts
P_nom = 4.0 # Watts
f = 2.0     # factor de servicio
C_nom_1 = 5.0
C_max_1 = 10.0
C_nom_2 = 15.0
C_max_2 = 30.0
C_nom_3 = 35.0
C_max_3 = 70.0
C_nom_4 = 65.0
C_max_4 = 130.0
C_nom_5 = 150.0
C_max_5 = 300.0

# Cálculo de velocidades nominales y máximas

n_max_1 = (9500.0 * f * P_max) / C_nom_1
n_nom_1 = (9500.0 * f * P_nom) / C_max_1
n_max_2 = (9500.0 * f * P_max) / C_nom_2
n_nom_2 = (9500.0 * f * P_nom) / C_max_2
n_max_3 = (9500.0 * f * P_max) / C_nom_3
n_nom_3 = (9500.0 * f * P_nom) / C_max_3
n_max_4 = (9500.0 * f * P_max) / C_nom_4
n_nom_4 = (9500.0 * f * P_nom) / C_max_4
n_max_5 = (9500.0 * f * P_max) / C_nom_5
n_nom_5 = (9500.0 * f * P_nom) / C_max_5   


print(f"n_nom_1: {n_nom_1:.2f}")
print(f"n_max_1: {n_max_1:.2f}")
print(f"n_nom_2: {n_nom_2:.2f}")
print(f"n_max_2: {n_max_2:.2f}")
print(f"n_nom_3: {n_nom_3:.2f}")
print(f"n_max_3: {n_max_3:.2f}")
print(f"n_nom_4: {n_nom_4:.2f}")
print(f"n_max_4: {n_max_4:.2f}")
print(f"n_nom_5: {n_nom_5:.2f}")
print(f"n_max_5: {n_max_5:.2f}")

T_d = 130.0 / 584.0 * (2.0 * np.pi * n_max_5 / 60.0)

print(f"T_d: {T_d:.4f}")