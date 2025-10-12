#-- coding: utf-8 --
"""
Modelo completo de fuerzas de neumáticos con curva de potencia del GA16DE
y análisis de marchas - Vehículo de tracción delantera
Autor: Carlos Caamaño
""" # Para estefany Bustos <3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# PARÁMETROS DEL VEHÍCULO Y TRANSMISIÓN
# -------------------------------
m = 1000.0   # masa [kg]
a = 0.92     # distancia CG -> eje delantero [m]
b = 1.53     # distancia CG -> eje trasero [m] (L = a + b)
h_cg = 0.45  # altura del centro de masa [m]
tf = 1.70    # trocha delantera [m]
tr = 1.70    # trocha trasera [m]
g = 9.81     # gravedad [m/s²]
μ = 1.1      # coeficiente de fricción neumático-pista

# Parámetros de transmisión (Nissan con GA16DE - estimados)
r_llanta = 0.28  # radio de llanta [m] (195/50R15 ≈ 0.28m)
rel_diferencial = 4.1  # relación de diferencial
rel_marchas = {
    1: 3.250,  # 1ra marcha
    2: 1.850,  # 2da marcha  
    3: 1.321,  # 3ra marcha (confirmar)
    4: 1.000,  # 4ta marcha
    5: 0.820   # 5ta marcha
}
marcha_actual = 3  # 3ra marcha según piloto

# -------------------------------
# CURVA DE POTENCIA DEL MOTOR GA16DE
# -------------------------------
# Datos característicos del GA16DE (110 hp @ 6000 rpm, 145 Nm @ 4000 rpm)
rpm_range = np.array([1000, 2000, 3000, 4000, 5000, 6000, 6500, 7000])
potencia_hp = np.array([35, 65, 90, 105, 110, 110, 108, 105])  # HP
torque_nm = np.array([85, 120, 140, 145, 140, 130, 125, 120])  # Nm

# Convertir a kW y ajustar curva
potencia_kw = potencia_hp * 0.7457  # hp to kW

# Función de interpolación para la curva de potencia
def potencia_motor(rpm):
    return np.interp(rpm, rpm_range, potencia_kw)

def torque_motor(rpm):
    return np.interp(rpm, rpm_range, torque_nm)

# -------------------------------
# CÁLCULO DE VELOCIDAD POR MARCHA Y RPM
# -------------------------------
def velocidad_por_rpm(rpm, marcha):
    """Calcula la velocidad del vehículo para un RPM y marcha dados"""
    rel_total = rel_marchas[marcha] * rel_diferencial
    v_ms = (rpm * 2 * np.pi * r_llanta) / (60 * rel_total)
    return v_ms * 3.6  # km/h

def rpm_por_velocidad(v_kmh, marcha):
    """Calcula el RPM para una velocidad y marcha dadas"""
    v_ms = v_kmh / 3.6
    rel_total = rel_marchas[marcha] * rel_diferencial
    rpm = (v_ms * 60 * rel_total) / (2 * np.pi * r_llanta)
    return rpm

# -------------------------------
# PARÁMETROS AERODINÁMICOS CALIBRADOS
# -------------------------------
rho = 1.225
A = 1.8
Crr = 0.015
Cd = 0.34  # Valor calibrado

# Velocidad máxima en 3ra a 6500 rpm
v_max_3ra = velocidad_por_rpm(6500, 3)
print(f"Velocidad máxima en 3ra a 6500 rpm: {v_max_3ra:.1f} km/h")

# -------------------------------
# FUNCIÓN DE FUERZAS DE RESISTENCIA
# -------------------------------
def calcular_resistencias(v_kmh):
    v_ms = v_kmh / 3.6
    Fa = 0.5 * rho * Cd * A * v_ms**2
    Fr = Crr * m * g
    return Fa, Fr, Fa + Fr

# -------------------------------
# ANÁLISIS DE ACELERACIÓN POR MARCHAS
# -------------------------------
def analizar_aceleracion(v_kmh, marcha, ax_g_target):
    """Analiza la aceleración en un punto específico"""
    
    # Calcular RPM actual
    rpm_actual = rpm_por_velocidad(v_kmh, marcha)
    P_disponible = potencia_motor(rpm_actual)
    T_disponible = torque_motor(rpm_actual)
    
    # Calcular fuerza máxima en ruedas (considerando transmisión)
    rel_total = rel_marchas[marcha] * rel_diferencial
    eficiencia_trans = 0.85  # eficiencia de transmisión
    F_trac_max_motor = (T_disponible * rel_total * eficiencia_trans) / r_llanta
    
    # Resistencia
    Fa, Fr, F_resist = calcular_resistencias(v_kmh)
    
    # Cargas normales (simplificado para análisis longitudinal)
    L = a + b
    FzF0 = m * g * b / L
    FzR0 = m * g * a / L
    ax = ax_g_target * g
    dF_long = m * ax * h_cg / L
    FzF = FzF0 - dF_long
    F_trac_max_adhesion = μ * FzF  # Solo ejes delanteros
    
    # Fuerza máxima disponible (mínimo entre motor y adherencia)
    F_trac_max = min(F_trac_max_motor, F_trac_max_adhesion)
    
    # Fuerza requerida
    F_total_required = m * ax + F_resist
    
    # Resultados
    return {
        'v_kmh': v_kmh,
        'rpm': rpm_actual,
        'marcha': marcha,
        'P_motor_kW': P_disponible,
        'T_motor_Nm': T_disponible,
        'F_trac_max_motor': F_trac_max_motor,
        'F_trac_max_adhesion': F_trac_max_adhesion,
        'F_resist': F_resist,
        'F_required': F_total_required,
        'ax_posible_g': (F_trac_max - F_resist) / m / g,
        'limite_motor': F_trac_max_motor < F_trac_max_adhesion,
        'limite_adhesion': F_trac_max_adhesion < F_trac_max_motor
    }

# -------------------------------
# ANÁLISIS EN EL PUNTO CRÍTICO (124 km/h, 3ra marcha)
# -------------------------------
print("="*80)
print("ANÁLISIS EN PUNTO CRÍTICO - 124 km/h, 3ra marcha")
print("="*80)

resultado = analizar_aceleracion(124, 3, 0.5)

print(f"RPM: {resultado['rpm']:.0f}")
print(f"Potencia motor: {resultado['P_motor_kW']:.1f} kW ({resultado['P_motor_kW']/0.7457:.1f} hp)")
print(f"Torque motor: {resultado['T_motor_Nm']:.1f} Nm")
print(f"Fuerza tracción máxima (motor): {resultado['F_trac_max_motor']:.1f} N")
print(f"Fuerza tracción máxima (adherencia): {resultado['F_trac_max_adhesion']:.1f} N")
print(f"Fuerza resistencia: {resultado['F_resist']:.1f} N")
print(f"Fuerza requerida para 0.5g: {resultado['F_required']:.1f} N")
print(f"Aceleración posible: {resultado['ax_posible_g']:.3f}g")

if resultado['limite_motor']:
    print("⚡ LÍMITE POR POTENCIA/MOTOR")
elif resultado['limite_adhesion']:
    print("🛞 LÍMITE POR ADHERENCIA")
else:
    print("✅ SIN LÍMITES")

# -------------------------------
# GRÁFICO DE CURVA DE POTENCIA Y TORQUE
# -------------------------------
plt.figure(figsize=(12, 6))

# Curva de potencia
plt.subplot(1, 2, 1)
plt.plot(rpm_range, potencia_kw, 'b-', linewidth=2, label='Potencia (kW)')
plt.plot(rpm_range, potencia_hp, 'r--', linewidth=2, label='Potencia (hp)')
plt.axvline(x=6500, color='gray', linestyle=':', alpha=0.7, label='6500 RPM (máx 3ra)')
plt.xlabel('RPM')
plt.ylabel('Potencia')
plt.title('Curva de Potencia - GA16DE')
plt.grid(True, alpha=0.3)
plt.legend()

# Curva de torque
plt.subplot(1, 2, 2)
plt.plot(rpm_range, torque_nm, 'g-', linewidth=2, label='Torque (Nm)')
plt.axvline(x=6500, color='gray', linestyle=':', alpha=0.7, label='6500 RPM')
plt.xlabel('RPM')
plt.ylabel('Torque (Nm)')
plt.title('Curva de Torque - GA16DE')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('curva_ga16de.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# ANÁLISIS DE MARCHAS COMPLETO
# -------------------------------
print("\n" + "="*80)
print("ANÁLISIS COMPARATIVO POR MARCHAS")
print("="*80)

marchas_analizar = [1, 2, 3, 4]
velocidades = np.linspace(50, 180, 14)  # De 50 a 180 km/h

resultados_completos = []
for marcha in marchas_analizar:
    for v in velocidades:
        rpm = rpm_por_velocidad(v, marcha)
        if 1000 <= rpm <= 7000:  # Rango válido del motor
            res = analizar_aceleracion(v, marcha, 0.5)
            resultados_completos.append(res)

# Crear DataFrame para análisis
df_analisis = pd.DataFrame(resultados_completos)

# Encontrar mejor marcha para cada velocidad
mejor_marcha = []
for v in velocidades:
    df_v = df_analisis[df_analisis['v_kmh'] == v]
    if len(df_v) > 0:
        mejor_idx = df_v['ax_posible_g'].idxmax()
        mejor_marcha.append(df_v.loc[mejor_idx])
    else:
        mejor_marcha.append(None)

print("Velocidad | Marcha | RPM | Acel.max (g) | Límite")
print("-"*50)
for res in mejor_marcha:
    if res is not None:
        limite = "Motor" if res['limite_motor'] else "Adherencia" if res['limite_adhesion'] else "None"
        print(f"{res['v_kmh']:6.0f} km/h | {res['marcha']:6} | {res['rpm']:4.0f} | {res['ax_posible_g']:6.3f}g | {limite}")

# -------------------------------
# RECOMENDACIONES FINALES
# -------------------------------
print("\n" + "="*80)
print("RECOMENDACIONES PARA EL PILOTO")
print("="*80)
print("1. ✅ 3ra marcha es óptima para 124 km/h")
print("2. ⚠️  Operación cerca del límite de adherencia")
print("3. 📈 Considerar ajuste de relación de marchas")
print("4. 🔧 Verificar estado de neumáticos y suspensión")
print("5. 🎯 Punto óptimo de cambio: ~6000 RPM")