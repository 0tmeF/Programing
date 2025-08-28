#-- coding: utf-8 --
"""
Modelo lineal de fuerzas de neumáticos por rueda con transferencia de carga
y limitación por fricción - Vehículo de tracción delantera
Autor: Carlos Caamaño
""" # Para estefany Bustos <3

import numpy as np
import pandas as pd

# -------------------------------
# PARÁMETROS DEL VEHÍCULO
# -------------------------------
m = 1000.0   # masa [kg]
a = 1.20     # distancia CG -> eje delantero [m]
b = 1.40     # distancia CG -> eje trasero [m] (L = a + b)
h_cg = 0.60  # altura del centro de masa [m]
tf = 1.58    # trocha delantera [m]
tr = 1.56    # trocha trasera [m]
g = 9.81     # gravedad [m/s²]
P_max = 75.0 # potencia máxima [kW] = 110 hp
v_max_potencia = 190.0 # velocidad máxima a potencia máxima [km/h]
v_max_05g = 124.0 # velocidad máxima a 0.5g aceleración [km/h]
μ = 1.1      # coeficiente de fricción neumático-pista

# -------------------------------
# CÁLCULO DE PARÁMETROS AERODINÁMICOS CALIBRADOS
# -------------------------------
rho = 1.225  # densidad del aire [kg/m³]
A = 1.8      # área frontal [m²]
Crr = 0.015  # coeficiente de resistencia a la rodadura

# Calcular Cd a partir de la velocidad máxima a potencia máxima
v_max_ms = v_max_potencia / 3.6
# En velocidad máxima: P_max = (Fa + Fr) * v_max_ms
# Donde Fa = 0.5*rho*Cd*A*v_max_ms² y Fr = Crr*m*g
# Resolviendo para Cd:
F_resist_max = P_max * 1000 / v_max_ms  # Fuerza total de resistencia a v_max
Fr_max = Crr * m * g  # Resistencia a rodadura
Fa_max = F_resist_max - Fr_max  # Resistencia aerodinámica
Cd_calculado = (2 * Fa_max) / (rho * A * v_max_ms**2)

# Verificar con el punto de 0.5g a 124 km/h
v_05g_ms = v_max_05g / 3.6
ax_05g = 0.5 * g
F_total_05g = m * ax_05g
Fa_05g = 0.5 * rho * Cd_calculado * A * v_05g_ms**2
Fr_05g = Crr * m * g
F_resist_05g = Fa_05g + Fr_05g
F_trac_required = F_total_05g + F_resist_05g

print(f"Coeficiente de arrastre calculado: Cd = {Cd_calculado:.3f}")
print(f"Fuerza de tracción requerida a 124 km/h y 0.5g: {F_trac_required:.1f} N")

# -------------------------------
# PARÁMETROS DE LOS NEUMÁTICOS
# -------------------------------
kK = 20.0       # rigidez longitudinal entre 14 y 20
k_alpha = 10.0  # rigidez lateral con ángulo de deriva [1/rad] entre 10 y 20
k_gamma = 1.0   # rigidez lateral con ángulo de caída [1/rad] entre 0.8 y 1.2

# -------------------------------
# ESTADOS DEL VEHÍCULO (ensayo a 124 km/h y 0.5g)
# -------------------------------
ax_g_target = +0.5    # aceleración longitudinal en g objetivo
ay_g = 0.0            # aceleración lateral en g
v = v_max_05g         # velocidad [km/h] - punto crítico de 0.5g

# Estados de cada rueda: FL, FR, RL, RR
K = (-0.10, -0.10, 0.0, 0.0)  # slip ratio aumentado para máxima tracción
alpha = (0.0, 0.0, 0.0, 0.0)  # ángulo de deriva [rad]
gamma = (-0.5*np.pi/180, -0.5*np.pi/180,
         -0.5*np.pi/180, -0.5*np.pi/180)  # camber [rad]

# -------------------------------
# CÁLCULO DE FUERZAS DE RESISTENCIA
# -------------------------------
v_ms = v / 3.6  # velocidad en m/s

# Fuerza aerodinámica (con Cd calculado)
Fa = 0.5 * rho * Cd_calculado * A * v_ms**2

# Fuerza de resistencia a la rodadura
Fr = Crr * m * g

# Fuerza total de resistencia
F_resist = Fa + Fr

# -------------------------------
# CÁLCULO DE CARGAS NORMALES
# -------------------------------
ax = ax_g_target * g
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
dF_lat_rear = (m * ay * h_cg) * (a/L) / tr

# Distribución entre ruedas
Fz_FL = max(0, FzF/2 - dF_lat_front/2)
Fz_FR = max(0, FzF/2 + dF_lat_front/2)
Fz_RL = max(0, FzR/2 - dF_lat_rear/2)
Fz_RR = max(0, FzR/2 + dF_lat_rear/2)

N = {"FL": Fz_FL, "FR": Fz_FR, "RL": Fz_RL, "RR": Fz_RR}

# -------------------------------
# FUERZAS POR RUEDA (CON LIMITACIÓN POR FRICCIÓN)
# -------------------------------
ruedas = ["FL", "FR", "RL", "RR"]
rows = []

# Fuerza total requerida para la aceleración
F_total_required = m * ax + F_resist

# Máxima fuerza de tracción disponible en ejes delanteros
F_trac_max = μ * (Fz_FL + Fz_FR)

print(f"\nFuerza total requerida: {F_total_required:.1f} N")
print(f"Fuerza máxima de tracción disponible: {F_trac_max:.1f} N")
print(f"Margen de tracción: {F_trac_max - F_total_required:.1f} N")

# Verificar si hay suficiente tracción
if F_total_required > F_trac_max:
    print("⚠️  ADVERTENCIA: La fuerza requerida excede la tracción disponible")
    ax_real = (F_trac_max - F_resist) / m
    print(f"   Aceleración real máxima: {ax_real/g:.3f}g")
    F_total = F_trac_max
else:
    F_total = F_total_required
    print("✅ Tracción suficiente disponible")

# Distribuir fuerza de tracción entre ruedas delanteras
if (Fz_FL + Fz_FR) > 0:
    factor_FL = Fz_FL / (Fz_FL + Fz_FR)
    factor_FR = Fz_FR / (Fz_FL + Fz_FR)
else:
    factor_FL, factor_FR = 0.5, 0.5

for i, w in enumerate(ruedas):
    Nw = N[w]
    
    # Fuerzas base del modelo lineal
    Xo_base = kK * Nw * K[i]
    Yo_base = k_alpha * Nw * alpha[i] + k_gamma * Nw * gamma[i]
    
    # Ajustar fuerzas longitudinales para ruedas motrices
    if w in ["FL", "FR"]:  # ruedas delanteras (motrices)
        if w == "FL":
            Xo = factor_FL * F_total
        else:
            Xo = factor_FR * F_total
    else:  # ruedas traseras
        Xo = 0.0
    
    # Limitar por círculo de fricción
    F_max = μ * Nw
    F_total_magnitude = np.sqrt(Xo**2 + Yo_base**2)
    
    if F_total_magnitude > F_max:
        scale_factor = F_max / F_total_magnitude
        Xo *= scale_factor
        Yo = Yo_base * scale_factor
        limited = True
    else:
        Yo = Yo_base
        limited = False
    
    rows.append({
        "Rueda": w,
        "N [N]": round(Nw, 1),
        "K [-]": K[i],
        "X [N]": round(Xo, 1),
        "Y [N]": round(Yo, 1),
        "F_max [N]": round(F_max, 1),
        "Utilización": f"{F_total_magnitude/F_max*100:.1f}%",
        "Limitado": limited
    })

# -------------------------------
# RESULTADOS
# -------------------------------
df = pd.DataFrame(rows)
print("\n" + "="*90)
print("ANÁLISIS DE FUERZAS EN NEUMÁTICOS - PUNTO CRÍTICO (124 km/h, 0.5g)")
print("="*90)
print(f"Velocidad: {v} km/h ({v_ms:.1f} m/s)")
print(f"Resistencia aerodinámica: {Fa:.1f} N (Cd = {Cd_calculado:.3f})")
print(f"Resistencia a la rodadura: {Fr:.1f} N")
print(f"Resistencia total: {F_resist:.1f} N")
print(f"Aceleración objetivo: {ax_g_target}g")
print(f"Aceleración real: {F_total/m/g:.3f}g")
print("="*90)
print(df.to_string(index=False))
print("="*90)

# Verificación de potencia
potencia_requerida = F_total * v_ms / 1000  # kW
print(f"\nPotencia requerida: {potencia_requerida:.1f} kW")
print(f"Potencia disponible: {P_max:.1f} kW")

if potencia_requerida > P_max:
    print("⚠️  ADVERTENCIA: La potencia requerida excede la disponible")
    print(f"   Déficit: {potencia_requerida - P_max:.1f} kW")
else:
    print(f"✅ Potencia suficiente (margen: {P_max - potencia_requerida:.1f} kW)")

# -------------------------------
# ANÁLISIS DE LÍMITE DE ADHERENCIA
# -------------------------------
print(f"\n{'='*50}")
print("ANÁLISIS DE LÍMITE DE ADHERENCIA")
print(f"{'='*50}")
print(f"Tracción requerida: {F_total_required:.1f} N")
print(f"Tracción disponible: {F_trac_max:.1f} N")
print(f"Margen de adherencia: {F_trac_max - F_total_required:.1f} N")
print(f"Porcentaje de utilización: {F_total_required/F_trac_max*100:.1f}%")

if F_total_required/F_trac_max > 0.95:
    print("⚠️  PELIGRO: Operación muy cerca del límite de adherencia")
elif F_total_required/F_trac_max > 0.85:
    print("📢 AVISO: Operación en régimen de alta exigencia")
else:
    print("✅ Régimen de operación seguro")