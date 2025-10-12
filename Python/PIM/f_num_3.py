# -*- coding: utf-8 -*-
"""
THERMODYNAMIC ANALYSIS OF TIRES - VERSION 3
Analysis of power, longitudinal slip and slip ratio limits
Equipo: [Nombre del equipo]
Autor: Carlos Caamaño
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# -------------------------------
# PARÁMETROS DEL VEHÍCULO Y NEUMÁTICOS
# -------------------------------
VEHICLE_PARAMS = {
    'mass': 1000.0,      # Masa [kg]
    'gravity': 9.81,     # Gravedad [m/s²]
    'a': 1.20,           # Distancia CG -> eje delantero [m]
    'b': 1.40,           # Distancia CG -> eje trasero [m]
    'h_cg': 0.60,        # Altura del centro de masa [m]
    'track_front': 1.58, # Trocha delantera [m]
    'track_rear': 1.56   # Trocha trasera [m]
}

TIRE_PARAMS = {
    'k_K': 18.0,         # Longitudinal stiffness [14-20]
    'k_alpha': 12.0,     # Lateral stiffness [10-20]
    'k_gamma': 1.1,      # Camber stiffness [0.8-1.2]
    'mu_max': 1.4        # Maximum friction coefficient
}

SLIP_LIMITS = {
    'K_opt': 0.10,       # Optimal slip ratio (10%)
    'K_max': 0.20,       # Maximum slip ratio (20%)
    'K_min': 0.02        # Minimum slip ratio for traction (2%)
}

# -------------------------------
# DATOS EXPERIMENTALES (ejemplo - reemplazar con datos reales)
# -------------------------------
race_data = {
    'time': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'speed_kmh': [30, 45, 60, 75, 90, 105, 120, 110, 95, 80, 65],
    'ax_g': [0.3, 0.5, 0.7, 0.9, 1.1, 0.9, 0.6, 0.4, 0.3, 0.2, 0.1],
    'ay_g': [0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
    'rpm': [3500, 4500, 5500, 6500, 7500, 7000, 6500, 6000, 5500, 5000, 4500],
    'gear': [2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 2]
}

# -------------------------------
# CARGA DE DATOS DESDE CSV
# -------------------------------
def load_race_data(csv_path):
    df = pd.read_csv(csv_path)
    print("Columnas disponibles en el CSV:")
    print(df.columns.tolist())
    race_data = {
        'time': df['loggingSample(N)'].tolist(),
        'speed_kmh': (df['locationSpeed(m/s)'] * 3.6).tolist(),  # <-- CAMBIO AQUÍ
        'ax_g': df['accelerometerAccelerationX(G)'].tolist(),
        'ay_g': df['accelerometerAccelerationY(G)'].tolist(),
        'rpm': df['rpm'].tolist() if 'rpm' in df.columns else [0]*len(df),  # Si no hay rpm, rellena con ceros
        'gear': df['gear'].tolist() if 'gear' in df.columns else [0]*len(df), # Si no hay gear, rellena con ceros
    }
    return race_data

# -------------------------------
# MODELO DE POTENCIA DEL MOTOR (GA16DE aproximado)
# -------------------------------
def engine_power(rpm: float) -> float:
    """
    Estimated power curve for GA16DE engine.
    """
    rpm_points = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    power_kw = [20, 40, 55, 65, 70, 75, 72, 65]  # kW
    return np.interp(rpm, rpm_points, power_kw)

# -------------------------------
# CÁLCULO DE CARGAS NORMALES
# -------------------------------
def calculate_normal_loads(ax_g: float, ay_g: float, params: dict) -> dict:
    """
    Calculates normal loads on each wheel.
    """
    m = params['mass']
    g = params['gravity']
    a = params['a']
    b = params['b']
    h_cg = params['h_cg']
    track_front = params['track_front']
    track_rear = params['track_rear']
    L = a + b

    # Cargas estáticas
    FzF0 = m * g * b / L
    FzR0 = m * g * a / L

    # Transferencia longitudinal
    dF_long = m * ax_g * g * h_cg / L
    FzF = FzF0 - dF_long
    FzR = FzR0 + dF_long

    # Transferencia lateral
    dF_lat_front = (m * ay_g * g * h_cg) * (b / L) / track_front
    dF_lat_rear = (m * ay_g * g * h_cg) * (a / L) / track_rear

    # Distribución entre ruedas
    Fz_FL = max(0, FzF / 2 - dF_lat_front / 2)
    Fz_FR = max(0, FzF / 2 + dF_lat_front / 2)
    Fz_RL = max(0, FzR / 2 - dF_lat_rear / 2)
    Fz_RR = max(0, FzR / 2 + dF_lat_rear / 2)

    return {
        'FL': Fz_FL, 'FR': Fz_FR,
        'RL': Fz_RL, 'RR': Fz_RR,
        'total': FzF + FzR,
        'front': Fz_FL + Fz_FR,
        'rear': Fz_RL + Fz_RR
    }

# -------------------------------
# CÁLCULO DE RESISTENCIAS
# -------------------------------
def calculate_resistances(speed_kmh: float, params: dict) -> float:
    """
    Calculates resistance forces (simplified).
    """
    m = params['mass']
    g = params['gravity']
    speed_ms = speed_kmh / 3.6
    # Aerodynamic resistance (simplified)
    Fa = 0.5 * 1.225 * 0.35 * 1.8 * speed_ms ** 2
    # Rolling resistance
    Fr = 0.015 * m * g
    return Fa + Fr

# -------------------------------
# ANÁLISIS DE POTENCIA Y SLIP RATIO
# -------------------------------
def analyze_power_slip(data: dict, vehicle_params: dict, tire_params: dict, slip_limits: dict) -> pd.DataFrame:
    """
    Analyzes available and required power, and calculates slip ratio.
    """
    results = []

    for i in range(len(data['time'])):
        # Datos del instante actual
        speed_kmh = data['speed_kmh'][i]
        speed_ms = speed_kmh / 3.6
        ax_g = data['ax_g'][i]
        ay_g = data['ay_g'][i]
        rpm = data['rpm'][i]
        gear = data['gear'][i]

        # 1. Calcular cargas normales
        loads = calculate_normal_loads(ax_g, ay_g, vehicle_params)
        N_total = loads['total']
        N_front = loads['front']

        # 2. Calcular fuerzas requeridas
        F_resistance = calculate_resistances(speed_kmh, vehicle_params)
        F_acceleration = vehicle_params['mass'] * ax_g * vehicle_params['gravity']
        F_total_required = F_acceleration + F_resistance

        # 3. Potencia requerida
        P_required = F_total_required * speed_ms / 1000  # kW

        # 4. Potencia disponible del motor
        P_available = engine_power(rpm)

        # 5. Fuerza de tracción máxima disponible
        F_trac_max = tire_params['mu_max'] * N_front  # Solo eje delantero motriz

        # 6. Calcular slip ratio (K) de la ecuación fundamental
        F_trac_actual = min(F_total_required, F_trac_max)
        mu_actual = F_trac_actual / N_front if N_front > 0 else 0
        K_actual = mu_actual / tire_params['k_K'] if tire_params['k_K'] > 0 else 0

        # 7. Verificar límites de slip ratio
        slip_state = "Óptimo"
        if K_actual > slip_limits['K_max']:
            slip_state = "Deslizamiento"
        elif K_actual < slip_limits['K_min']:
            slip_state = "Poco eficiente"
        elif abs(K_actual - slip_limits['K_opt']) > 0.03:
            slip_state = "Sub-óptimo"

        # 8. Calcular fuerza longitudinal (X_o)
        X_o = tire_params['k_K'] * N_front * K_actual

        results.append({
            'time': data['time'][i],
            'speed_kmh': speed_kmh,
            'rpm': rpm,
            'gear': gear,
            'ax_g': ax_g,
            'ay_g': ay_g,
            'N_total_N': round(N_total, 1),
            'N_front_N': round(N_front, 1),
            'F_total_req_N': round(F_total_required, 1),
            'F_trac_max_N': round(F_trac_max, 1),
            'P_req_kW': round(P_required, 1),
            'P_avail_kW': round(P_available, 1),
            'mu_actual': round(mu_actual, 3),
            'K_actual': round(K_actual, 3),
            'X_o_N': round(X_o, 1),
            'slip_state': slip_state,
            'power_margin': round(P_available - P_required, 1)
        })

    return pd.DataFrame(results)

# -------------------------------
# VISUALIZACIÓN DE RESULTADOS
# -------------------------------
def visualize_full_analysis(df: pd.DataFrame, slip_limits: dict, tire_params: dict) -> None:
    """
    Crea gráficos completos del análisis.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Gráfico 1: Potencia y Slip Ratio
    ax1.plot(df['time'], df['P_avail_kW'], 'b-', label='Potencia Disponible', linewidth=2)
    ax1.plot(df['time'], df['P_req_kW'], 'r-', label='Potencia Requerida', linewidth=2)
    ax1.set_xlabel('Tiempo [s]')
    ax1.set_ylabel('Potencia [kW]')
    ax1.set_title('Potencia Disponible vs Requerida')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax1_twin = ax1.twinx()
    ax1_twin.plot(df['time'], df['K_actual'] * 100, 'g--', label='Slip Ratio (%)', linewidth=2)
    ax1_twin.set_ylabel('Slip Ratio [%]', color='green')
    ax1_twin.tick_params(axis='y', labelcolor='green')
    ax1_twin.legend(loc='upper right')

    # Gráfico 2: Fuerzas y Cargas
    ax2.plot(df['time'], df['F_total_req_N'], 'r-', label='Fuerza Requerida', linewidth=2)
    ax2.plot(df['time'], df['F_trac_max_N'], 'b-', label='Fuerza Máxima', linewidth=2)
    ax2.plot(df['time'], df['X_o_N'], 'g--', label='X_o (Fuerza Longitudinal)', linewidth=2)
    ax2.set_xlabel('Tiempo [s]')
    ax2.set_ylabel('Fuerza [N]')
    ax2.set_title('Fuerzas de Tracción')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Gráfico 3: Slip Ratio y Límites
    ax3.plot(df['time'], df['K_actual'] * 100, 'b-', label='Slip Ratio Actual', linewidth=2)
    ax3.axhline(y=slip_limits['K_opt'] * 100, color='green', linestyle='--', label='Óptimo (10%)', alpha=0.7)
    ax3.axhline(y=slip_limits['K_max'] * 100, color='red', linestyle='--', label='Límite (20%)', alpha=0.7)
    ax3.axhline(y=slip_limits['K_min'] * 100, color='orange', linestyle='--', label='Mínimo (2%)', alpha=0.7)
    ax3.fill_between(df['time'], slip_limits['K_max'] * 100, 25, color='red', alpha=0.1, label='Zona de Deslizamiento')
    ax3.fill_between(df['time'], slip_limits['K_min'] * 100, 0, color='yellow', alpha=0.1, label='Zona Ineficiente')
    ax3.set_xlabel('Tiempo [s]')
    ax3.set_ylabel('Slip Ratio [%]')
    ax3.set_title('Análisis de Slip Ratio y Límites')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Gráfico 4: Coeficiente de Fricción
    ax4.plot(df['time'], df['mu_actual'], 'purple', label='μ Actual', linewidth=2)
    ax4.axhline(y=tire_params['mu_max'], color='red', linestyle='--', label='μ Máximo', alpha=0.7)
    ax4.set_xlabel('Tiempo [s]')
    ax4.set_ylabel('Coeficiente de Fricción (μ)')
    ax4.set_title('Coeficiente de Fricción Utilizado')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('power_slip_analysis_v3.png', dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------------
# ANÁLISIS ESTADÍSTICO
# -------------------------------
def statistical_slip_analysis(df: pd.DataFrame, slip_limits: dict) -> None:
    """
    Realiza análisis estadístico del slip ratio.
    """
    print("=" * 60)
    print("ANÁLISIS ESTADÍSTICO - SLIP RATIO")
    print("=" * 60)

    # Estadísticas básicas
    K_values = df['K_actual']
    print(f"Slip Ratio promedio: {np.mean(K_values) * 100:.1f}%")
    print(f"Slip Ratio máximo: {np.max(K_values) * 100:.1f}%")
    print(f"Slip Ratio mínimo: {np.min(K_values) * 100:.1f}%")
    print(f"Desviación estándar: {np.std(K_values) * 100:.1f}%")

    # Porcentaje de tiempo en cada zona
    total_time = len(df)
    optimal_time = len(df[abs(df['K_actual'] - slip_limits['K_opt']) < 0.03])
    slip_time = len(df[df['K_actual'] > slip_limits['K_max']])
    inefficient_time = len(df[df['K_actual'] < slip_limits['K_min']])

    print(f"\nDISTRIBUCIÓN TEMPORAL:")
    print(f"Zona óptima (8-12%): {optimal_time / total_time * 100:.1f}%")
    print(f"Zona deslizamiento (>20%): {slip_time / total_time * 100:.1f}%")
    print(f"Zona ineficiente (<2%): {inefficient_time / total_time * 100:.1f}%")
    print(f"Zona sub-óptima: {(100 - (optimal_time + slip_time + inefficient_time) / total_time * 100):.1f}%")

    # Correlación con otras variables
    corr_speed = np.corrcoef(df['speed_kmh'], df['K_actual'])[0, 1]
    corr_power = np.corrcoef(df['P_req_kW'], df['K_actual'])[0, 1]

    print(f"\nCORRELACIONES:")
    print(f"Con velocidad: {corr_speed:.3f}")
    print(f"Con potencia requerida: {corr_power:.3f}")

# -------------------------------
# EJECUCIÓN PRINCIPAL
# -------------------------------
if __name__ == "__main__":
    print("ANÁLISIS DE POTENCIA Y SLIP RATIO - VERSIÓN 3")
    print("=" * 50)

    # Solicitar ruta del archivo CSV
    csv_path = input("Ruta del archivo CSV a analizar: ").strip()
    if not os.path.exists(csv_path):
        print(f"Archivo CSV no encontrado: {csv_path}")
        exit(1)

    # Cargar datos reales
    race_data = load_race_data(csv_path)

    # Ejecutar análisis completo
    df_results = analyze_power_slip(race_data, VEHICLE_PARAMS, TIRE_PARAMS, SLIP_LIMITS)

    # Mostrar resultados principales
    print("\nRESULTADOS PRINCIPALES:")
    print("=" * 40)
    print(df_results[['time', 'speed_kmh', 'K_actual', 'slip_state', 'power_margin']].to_string(index=False))

    # Análisis estadístico
    statistical_slip_analysis(df_results, SLIP_LIMITS)

    # Visualización
    visualize_full_analysis(df_results, SLIP_LIMITS, TIRE_PARAMS)

    # Guardar resultados
    df_results.to_csv('power_slip_analysis_v3.csv', index=False)
    print(f"\nResultados guardados en 'power_slip_analysis_v3.csv'")