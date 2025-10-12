#-- coding: utf-8 --
"""
Modelo lineal de fuerzas de neumáticos por rueda con transferencia de carga
Autor: Carlos Caamaño
""" # Para estefany Bustos <3

import numpy as np
import pandas as pd

# -------------------------------
# PARÁMETROS DEL VEHÍCULO
# -------------------------------
m     = 1000.0   # masa [kg]
a     = 1.20     # distancia CG -> eje delantero [m]
b     = 1.40     # distancia CG -> eje trasero [m] (L = a + b)
h_cg  = 0.60     # altura del centro de masa [m]
tf    = 1.58     # trocha delantera [m]
tr    = 1.56     # trocha trasera [m]
g     = 9.81     # gravedad [m/s²]

# -------------------------------
# PARÁMETROS DE LOS NEUMÁTICOS
# -------------------------------
kK      = 20.0    # rigidez longitudinal entre 14 y 20
k_alpha = 10.0    # rigidez lateral con ángulo de deriva [1/rad] entre 10 y 20
k_gamma = 1.0     # rigidez lateral con ángulo de caída [1/rad] entre 0.8 y 1.2

# -------------------------------
# ESTADOS DEL VEHÍCULO (ensayo)
# -------------------------------
ax_g = +0.5   # aceleración longitudinal en g (negativo = frenado)
ay_g = 0.0   # aceleración lateral en g (positivo = giro izquierda)

# Estados de cada rueda: FL, FR, RL, RR
K     = ( -0.01, -0.01, -0.01, -0.01 )          # slip ratio
alpha = ( 0.0, 0.0, 0.0, 0.0 )          # ángulo de deriva [rad]
gamma = ( -0.5*np.pi/180, -0.5*np.pi/180,
          -0.5*np.pi/180, -0.5*np.pi/180 )      # camber [rad]

# -------------------------------
# CÁLCULO DE CARGAS NORMALES
# -------------------------------
# Pasamos a m/s²
ax = ax_g * g
ay = ay_g * g

L = a + b

# Carga estática por eje
FzF0 = m * g * b / L
FzR0 = m * g * a / L

# Transferencia longitudinal
dF_long = m * ax * h_cg / L
FzF = FzF0 - dF_long
FzR = FzR0 + dF_long

# Transferencia lateral
dF_lat_front = (m * ay * h_cg) * (b/L) / tf
dF_lat_rear  = (m * ay * h_cg) * (a/L) / tr

# Distribución entre ruedas
Fz_FL = FzF/2 - dF_lat_front/2
Fz_FR = FzF/2 + dF_lat_front/2
Fz_RL = FzR/2 - dF_lat_rear/2
Fz_RR = FzR/2 + dF_lat_rear/2

N = {"FL":Fz_FL, "FR":Fz_FR, "RL":Fz_RL, "RR":Fz_RR}

# -------------------------------
# FUERZAS POR RUEDA
# -------------------------------
ruedas = ["FL", "FR", "RL", "RR"]
rows = []
for i, w in enumerate(ruedas):
    Nw = N[w]
    Xo = kK * Nw * K[i]
    Yo = k_alpha * Nw * alpha[i] + k_gamma * Nw * gamma[i]
    rows.append({
        "Rueda": w,
        "N [N]": round(Nw,1),
        "K [-]": K[i],
        "alpha [rad]": alpha[i],
        "gamma [rad]": gamma[i],
        "X_o [N]": round(Xo,1),
        "Y_o [N]": round(Yo,1)
    })

df = pd.DataFrame(rows)
print(df)