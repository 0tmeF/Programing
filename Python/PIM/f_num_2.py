# -*- coding: utf-8 -*-
"""
Análisis térmico de neumáticos de competición
---------------------------------------------
Este script modela la temperatura ideal de trabajo de neumáticos de competición a partir de datos experimentales.

Autor: Carlos Caamaño
Equipo: [Team Name]
Fecha: 2025-08-28
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------------------
# PARÁMETROS CONSTANTES DEL VEHÍCULO
# -------------------------------
MASS = 1000.0      # Masa del vehículo [kg]
GRAVITY = 9.81     # Gravedad [m/s²]
A = 0.92           # Distancia CG al eje delantero [m]
B = 1.53           # Distancia CG al eje trasero [m]
H_CG = 0.45        # Altura del centro de gravedad [m]
TRACK_FRONT = 1.70 # Trocha delantera [m]
TRACK_REAR = 1.70  # Trocha trasera [m]

# -------------------------------
# DATOS DE PISTA DE EJEMPLO (reemplazar por datos reales)
# -------------------------------
track_data = {
    'time': [0, 1, 2, 3, 4],
    'ax_g': [0.1, 0.2, 0.4, 0.5, 0.5],
    'ay_g': [0, 0, 0, 0, 0.1]
}

# -------------------------------
# MODELO DE COEFICIENTE DE FRICCIÓN VS TEMPERATURA
# -------------------------------
def mu_vs_temperature(T: np.ndarray, mu_max: float, T_opt: float, width: float) -> np.ndarray:
    """
    Modelo tipo campana para el coeficiente de fricción en función de la temperatura.
    """
    return mu_max * np.exp(-((T - T_opt) / width) ** 2)

# -------------------------------
# CÁLCULO DE FUERZAS EN CADA NEUMÁTICO
# -------------------------------
def tire_forces(ax_g: float, ay_g: float, mu: float) -> tuple:
    """
    Calcula las fuerzas normales y de fricción en cada neumático considerando transferencia de carga.
    """
    ax = ax_g * GRAVITY
    ay = ay_g * GRAVITY
    L = A + B

    # Carga normal estática por eje
    FzF0 = MASS * GRAVITY * B / L
    FzR0 = MASS * GRAVITY * A / L

    # Transferencia longitudinal
    dF_long = MASS * ax * H_CG / L
    FzF = FzF0 - dF_long
    FzR = FzR0 + dF_long

    # Transferencia lateral
    dF_lat_front = (MASS * ay * H_CG) * (B / L) / TRACK_FRONT
    dF_lat_rear = (MASS * ay * H_CG) * (A / L) / TRACK_REAR

    # Distribución por rueda
    Fz_FL = max(0, FzF / 2 - dF_lat_front / 2)
    Fz_FR = max(0, FzF / 2 + dF_lat_front / 2)
    Fz_RL = max(0, FzR / 2 - dF_lat_rear / 2)
    Fz_RR = max(0, FzR / 2 + dF_lat_rear / 2)

    # Fuerzas de fricción
    friction_forces = {
        'FL': mu * Fz_FL,
        'FR': mu * Fz_FR,
        'RL': mu * Fz_RL,
        'RR': mu * Fz_RR
    }

    return Fz_FL, Fz_FR, Fz_RL, Fz_RR, friction_forces

# -------------------------------
# ANÁLISIS DE DATOS DE PISTA
# -------------------------------
def analyze_track_data(track_data: dict, mu: float) -> pd.DataFrame:
    """
    Analiza los datos de pista y calcula las fuerzas por neumático.
    """
    results = []
    for i in range(len(track_data['time'])):
        ax_g = track_data['ax_g'][i]
        ay_g = track_data['ay_g'][i]
        Fz_FL, Fz_FR, Fz_RL, Fz_RR, friction = tire_forces(ax_g, ay_g, mu)
        results.append({
            'time': track_data['time'][i],
            'ax_g': ax_g,
            'ay_g': ay_g,
            'Fz_FL': round(Fz_FL, 2),
            'Fz_FR': round(Fz_FR, 2),
            'Fz_RL': round(Fz_RL, 2),
            'Fz_RR': round(Fz_RR, 2),
            'F_fric_FL': round(friction['FL'], 2),
            'F_fric_FR': round(friction['FR'], 2),
            'F_fric_RL': round(friction['RL'], 2),
            'F_fric_RR': round(friction['RR'], 2),
        })
    return pd.DataFrame(results)

# -------------------------------
# SIMULACIÓN DE ENSAYO ESTÁTICO
# -------------------------------
def static_test_simulation() -> tuple:
    """
    Simula un ensayo estático de neumáticos a diferentes temperaturas.
    """
    estimated_params = {
        'mu_max': 1.4,
        'T_opt': 85.0,
        'width': 25.0
    }
    temperatures = np.linspace(20, 120, 21)
    mu_values = mu_vs_temperature(temperatures, **estimated_params)
    return temperatures, mu_values, estimated_params

# -------------------------------
# AJUSTE DE CURVA A DATOS EXPERIMENTALES
# -------------------------------
def fit_mu_curve(T_exp: np.ndarray, mu_exp: np.ndarray) -> tuple:
    """
    Ajusta la curva μ vs T a datos experimentales.
    """
    try:
        p0 = [max(mu_exp), np.mean(T_exp), 25.0]
        popt, pcov = curve_fit(
            mu_vs_temperature,
            T_exp,
            mu_exp,
            p0=p0,
            bounds=([0.8, 50, 10], [2.0, 120, 50])
        )
        fitted_params = {
            'mu_max': popt[0],
            'T_opt': popt[1],
            'width': popt[2]
        }
        return fitted_params, np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"Error en el ajuste de curva: {e}")
        return None, None

# -------------------------------
# PROTOCOLO DE ANÁLISIS COMPLETO
# -------------------------------
def full_analysis_protocol() -> tuple:
    """
    Protocolo completo para el análisis térmico de neumáticos.
    """
    print("PROTOCOLO DE ANÁLISIS TÉRMICO DE NEUMÁTICOS")
    print("=" * 60)

    # 1. Simulación de ensayo estático (reemplazar por datos reales)
    print("\n1. CURVA μ vs TEMPERATURA (Estimación inicial)")
    temperatures, mu_values, params = static_test_simulation()

    # 2. Análisis de datos de pista
    print("\n2. ANÁLISIS DE DATOS DE PISTA")
    df_track = analyze_track_data(track_data, params['mu_max'])

    # 3. Resultados principales
    print("\n3. RESULTADOS PRINCIPALES")
    print(f"Temperatura óptima estimada: {params['T_opt']}°C")
    print(f"μ máximo estimado: {params['mu_max']:.3f}")
    print(f"Máxima fuerza de fricción delantera: {df_track[['F_fric_FL', 'F_fric_FR']].max().max():.1f} N")
    print(f"Máxima fuerza de fricción trasera: {df_track[['F_fric_RL', 'F_fric_RR']].max().max():.1f} N")

    return temperatures, mu_values, df_track, params

# -------------------------------
# VISUALIZACIÓN DE RESULTADOS
# -------------------------------
def visualize_results(temperatures: np.ndarray, mu_values: np.ndarray, df_track: pd.DataFrame, params: dict) -> None:
    """
    Genera gráficos para el informe del equipo.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Gráfico 1: μ vs Temperatura
    ax1.plot(temperatures, mu_values, 'b-', linewidth=2)
    ax1.axvline(params['T_opt'], color='r', linestyle='--', alpha=0.7, label=f'T óptima: {params["T_opt"]}°C')
    ax1.set_xlabel('Temperatura [°C]')
    ax1.set_ylabel('Coeficiente de fricción (μ)')
    ax1.set_title('μ vs Temperatura - Neumáticos Semi-Slick')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Gráfico 2: Fuerzas G en pista
    ax2.plot(df_track['time'], df_track['ax_g'], 'r-', label='Aceleración Longitudinal (g)')
    ax2.plot(df_track['time'], df_track['ay_g'], 'b-', label='Aceleración Lateral (g)')
    ax2.set_xlabel('Tiempo [s]')
    ax2.set_ylabel('Aceleración [g]')
    ax2.set_title('Datos de pista - Fuerzas G')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Gráfico 3: Cargas normales
    ax3.plot(df_track['time'], df_track['Fz_FL'], label='FL')
    ax3.plot(df_track['time'], df_track['Fz_FR'], label='FR')
    ax3.plot(df_track['time'], df_track['Fz_RL'], label='RL')
    ax3.plot(df_track['time'], df_track['Fz_RR'], label='RR')
    ax3.set_xlabel('Tiempo [s]')
    ax3.set_ylabel('Carga normal [N]')
    ax3.set_title('Distribución de carga normal por neumático')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Gráfico 4: Fuerzas de fricción
    ax4.plot(df_track['time'], df_track['F_fric_FL'], label='FL')
    ax4.plot(df_track['time'], df_track['F_fric_FR'], label='FR')
    ax4.plot(df_track['time'], df_track['F_fric_RL'], label='RL')
    ax4.plot(df_track['time'], df_track['F_fric_RR'], label='RR')
    ax4.set_xlabel('Tiempo [s]')
    ax4.set_ylabel('Fuerza de fricción [N]')
    ax4.set_title('Fuerzas de fricción por neumático')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('analisis_termico_neumaticos.png', dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------------
# EJECUCIÓN PRINCIPAL
# -------------------------------
if __name__ == "__main__":
    # Ejecutar análisis completo
    temperatures, mu_values, df_track, params = full_analysis_protocol()

    # Mostrar tabla de resultados
    print("\n4. TABLA DE RESULTADOS (primeras 5 filas)")
    print(df_track.head().to_string(index=False))

    # Generar gráficos
    visualize_results(temperatures, mu_values, df_track, params)

    print("\n5. RECOMENDACIONES PARA ENSAYO DE LABORATORIO")
    print("   - Medir μ en: 60°C, 70°C, 80°C, 90°C, 100°C")
    print("   - Al menos 3 mediciones por temperatura")
    print("   - Mantener constante la presión de inflado")
    print("   - Permitir tiempo de estabilización térmica")