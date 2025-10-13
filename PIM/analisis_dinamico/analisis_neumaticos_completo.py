# -*- coding: utf-8 -*-
"""
Análisis térmico y de fuerzas en neumáticos de competición
----------------------------------------------------------
Este script modela la temperatura ideal de trabajo y fuerzas en neumáticos de competición 
a partir de datos experimentales, incluyendo modelo lineal de fuerzas por rueda.

Autor: Carlos Caamaño C
Equipo: [Team Name]
Fecha: 2025-09-23

Ruta de acceso archivo csv:
---------------------------
/Users/carlos/Documents/UdeC/2025/Segundo semetre/PIM/Datos/Datos Carrera 21:09:2025/primera_manga_filtrada.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# -------------------------------
# PARÁMETROS CONSTANTES DEL VEHÍCULO
# -------------------------------
M = 1000.0         # Masa del vehículo [kg]
g = 9.81           # Gravedad [m/s²]
a = 0.92           # Distancia CG al eje delantero [m]
b = 1.53           # Distancia CG al eje trasero [m]
h_cg = 0.45        # Altura CG [m]
track_front = 1.70 # Trocha delantera [m]
track_rear = 1.70  # Trocha trasera [m]
mu = 1.0           # Coeficiente de fricción (valor dado)

# -------------------------------
# PARÁMETROS DE LOS NEUMÁTICOS (Nuevos - de f_neumaticos.py)
# -------------------------------
kK = 20.0          # Rigidez longitudinal entre 14 y 20 [adimensional]
k_alpha = 10.0     # Rigidez lateral con ángulo de deriva [1/rad] entre 10 y 20
k_gamma = 1.0      # Rigidez lateral con ángulo de caída [1/rad] entre 0.8 y 1.2

# -------------------------------
# FUNCIONES DE ANÁLISIS Y UTILIDAD
# -------------------------------

def tire_forces(ax_g: float, ay_g: float, az_g: float, mu: float) -> tuple:
    """
    Calcula las fuerzas normales y de fricción en cada neumático considerando transferencia de carga y aceleración vertical.
    """
    ax = ax_g * g
    ay = ay_g * g
    az = az_g * g
    w = a + b

    # Carga normal estática por eje, ajustada por aceleración vertical
    FzF0 = M * (g) * b / w
    FzR0 = M * (g) * a / w

    # Transferencia longitudinal
    dF_long = M * ax * h_cg / w
    FzF = FzF0 - dF_long
    FzR = FzR0 + dF_long

    # Transferencia lateral
    dF_lat_front = (M * ay * h_cg) * (b / w) / track_front
    dF_lat_rear = (M * ay * h_cg) * (a / w) / track_rear

    # Distribución por rueda
    Fz_FL = max(0, FzF / 2 - dF_lat_front / 2)
    Fz_FR = max(0, FzF / 2 + dF_lat_front / 2)
    Fz_RL = max(0, FzR / 2 - dF_lat_rear / 2)
    Fz_RR = max(0, FzR / 2 + dF_lat_rear / 2)

    friction_forces = {
        'FL': mu * Fz_FL,
        'FR': mu * Fz_FR,
        'RL': mu * Fz_RL,
        'RR': mu * Fz_RR
    }

    return Fz_FL, Fz_FR, Fz_RL, Fz_RR, friction_forces

def calculate_tire_forces_linear(Fz_FL: float, Fz_FR: float, Fz_RL: float, Fz_RR: float, 
                                K: tuple = None, alpha: tuple = None, gamma: tuple = None) -> pd.DataFrame:
    """
    Calcula fuerzas longitudinales y laterales usando modelo lineal de neumáticos.
    (Nueva función de f_neumaticos.py)
    
    Parameters:
    - Fz_XX: Cargas normales por rueda [N]
    - K: Slip ratio por rueda (default: -0.01 para todas)
    - alpha: Ángulo de deriva por rueda [rad] (default: 0 para todas)
    - gamma: Ángulo de camber por rueda [rad] (default: -0.5° para todas)
    """
    if K is None:
        K = (-0.01, -0.01, -0.01, -0.01)
    if alpha is None:
        alpha = (0.0, 0.0, 0.0, 0.0)
    if gamma is None:
        gamma = (-0.5*np.pi/180, -0.5*np.pi/180, -0.5*np.pi/180, -0.5*np.pi/180)
    
    ruedas = ["FL", "FR", "RL", "RR"]
    Fz_values = [Fz_FL, Fz_FR, Fz_RL, Fz_RR]
    
    rows = []
    for i, w in enumerate(ruedas):
        Nw = Fz_values[i]
        # Fuerza longitudinal: Xo = kK * N * K
        Xo = kK * Nw * K[i]
        # Fuerza lateral: Yo = k_alpha * N * alpha + k_gamma * N * gamma
        Yo = k_alpha * Nw * alpha[i] + k_gamma * Nw * gamma[i]
        
        rows.append({
            "Rueda": w,
            "N [N]": round(Nw, 1),
            "K [-]": K[i],
            "alpha [rad]": alpha[i],
            "gamma [rad]": gamma[i],
            "X_o [N]": round(Xo, 1),
            "Y_o [N]": round(Yo, 1),
            "Fuerza_Total [N]": round(np.sqrt(Xo**2 + Yo**2), 1)
        })
    
    return pd.DataFrame(rows)

def analyze_track_data(track_data: dict, mu: float, ax_offset: float = -0.080, 
                      ay_offset: float = -0.090, az_offset: float = -0.1) -> pd.DataFrame:
    """
    Analiza los datos de pista y calcula las fuerzas por neumático.
    """
    results = []
    for i in range(len(track_data['time'])):
        ax_g = track_data['ax_g'][i] + ax_offset
        ay_g = track_data['ay_g'][i] + ay_offset
        # Corrige el offset de gravedad del sensor
        az_g = track_data['az_g'][i] - az_offset
        
        Fz_FL, Fz_FR, Fz_RL, Fz_RR, friction = tire_forces(ax_g, ay_g, az_g, mu)
        
        # Calcular fuerzas de neumáticos con modelo lineal (punto específico en el tiempo)
        tire_forces_df = calculate_tire_forces_linear(Fz_FL, Fz_FR, Fz_RL, Fz_RR)
        
        results.append({
            'time': track_data['time'][i],
            'ax_g': ax_g,
            'ay_g': ay_g,
            'az_g': az_g,
            'Fz_FL': round(Fz_FL, 2),
            'Fz_FR': round(Fz_FR, 2),
            'Fz_RL': round(Fz_RL, 2),
            'Fz_RR': round(Fz_RR, 2),
            'F_fric_FL': round(friction['FL'], 2),
            'F_fric_FR': round(friction['FR'], 2),
            'F_fric_RL': round(friction['RL'], 2),
            'F_fric_RR': round(friction['RR'], 2),
            # Nuevos campos del modelo lineal
            'X_o_FL': tire_forces_df[tire_forces_df['Rueda'] == 'FL']['X_o [N]'].iloc[0],
            'Y_o_FL': tire_forces_df[tire_forces_df['Rueda'] == 'FL']['Y_o [N]'].iloc[0],
            'X_o_FR': tire_forces_df[tire_forces_df['Rueda'] == 'FR']['X_o [N]'].iloc[0],
            'Y_o_FR': tire_forces_df[tire_forces_df['Rueda'] == 'FR']['Y_o [N]'].iloc[0],
            'X_o_RL': tire_forces_df[tire_forces_df['Rueda'] == 'RL']['X_o [N]'].iloc[0],
            'Y_o_RL': tire_forces_df[tire_forces_df['Rueda'] == 'RL']['Y_o [N]'].iloc[0],
            'X_o_RR': tire_forces_df[tire_forces_df['Rueda'] == 'RR']['X_o [N]'].iloc[0],
            'Y_o_RR': tire_forces_df[tire_forces_df['Rueda'] == 'RR']['Y_o [N]'].iloc[0],
        })
    
    return pd.DataFrame(results)

def find_data_start(csv_path: str, threshold: float = 0.2, window: int = 2000) -> int:
    """
    Busca el primer índice donde las aceleraciones (X o Y) superan un umbral
    durante una ventana consecutiva de datos. Así se identifica el inicio útil.
    """
    df = pd.read_csv(csv_path)
    ax_col = 'accelerometerAccelerationX(G)'
    ay_col = 'accelerometerAccelerationY(G)'

    abs_acc = np.abs(df[ax_col]) + np.abs(df[ay_col])

    for i in range(len(df) - window):
        if np.all(abs_acc.iloc[i:i+window] > threshold):
            print(f"Primer dato útil en el índice {i}, tiempo {df['loggingSample(N)'].iloc[i]}")
            return i
    print("No se encontró un bloque de datos útiles.")
    return None

def load_and_clean_csv(csv_path: str, output_name: str, threshold=0.3, window=10, min_time=None):
    """
    Carga y limpia el archivo CSV, buscando el inicio útil de los datos.
    Guarda un nuevo archivo CSV con los datos limpios.
    """
    if not os.path.exists(csv_path):
        print(f"Archivo CSV no encontrado: {csv_path}")
        exit(1)
    df_raw = pd.read_csv(csv_path)
    time_col = 'loggingSample(N)'
    if min_time is not None:
        df_raw = df_raw[df_raw[time_col] >= min_time].reset_index(drop=True)
        print(f"Archivo cortado por tiempo mínimo ({min_time}), {len(df_raw)} muestras útiles.")
    else:
        start_idx = find_data_start(csv_path, threshold, window)
        if start_idx is not None:
            df_raw = df_raw.iloc[start_idx:].reset_index(drop=True)
            print(f"Archivo cortado por aceleración, {len(df_raw)} muestras útiles.")
        else:
            print("Se usará el archivo completo.")
    df_raw.to_csv(output_name, index=False)
    print(f"Archivo limpio guardado como '{output_name}'")
    ax_col = 'accelerometerAccelerationX(G)'  # Lateral
    ay_col = 'accelerometerAccelerationY(G)'  # Longitudinal
    az_col = 'accelerometerAccelerationZ(G)'  # Vertical
    track_data = {
        'time': df_raw[time_col].tolist(),
        'ax_g': df_raw[ax_col].tolist(),
        'ay_g': df_raw[ay_col].tolist(),
        'az_g': df_raw[az_col].tolist(),
    }
    print(f"Datos cargados: {len(track_data['time'])} muestras")
    return track_data

def visualize_results(df_track: pd.DataFrame, params: dict = None) -> None:
    """
    Genera gráficos en figuras separadas para el informe del equipo.
    Incluye nuevas visualizaciones del modelo lineal de neumáticos.
    """
    if params is None:
        params = {}

    # 1. Aceleración lateral (X)
    plt.figure(figsize=(10, 6))
    plt.plot(df_track['time'], df_track['ax_g'], 'b-', label='Aceleración Lateral (g)', linewidth=1)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Aceleración Lateral [g]')
    plt.title('Aceleración Lateral en pista')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Aceleración longitudinal (Y)
    plt.figure(figsize=(10, 6))
    plt.plot(df_track['time'], df_track['ay_g'], 'r-', label='Aceleración Longitudinal (g)', linewidth=1)
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Aceleración Longitudinal [g]')
    plt.title('Aceleración Longitudinal en pista')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Carga normal - individual por rueda
    ruedas = [
        ('Fz_FL', 'Front Left (FL)', 'blue'),
        ('Fz_FR', 'Front Right (FR)', 'red'),
        ('Fz_RL', 'Rear Left (RL)', 'green'),
        ('Fz_RR', 'Rear Right (RR)', 'orange')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, (col, label, color) in enumerate(ruedas):
        axes[idx].plot(df_track['time'], df_track[col], color=color, label=label, linewidth=1)
        axes[idx].set_xlabel('Tiempo [s]')
        axes[idx].set_ylabel('Carga normal [N]')
        axes[idx].set_title(f'Carga normal en {label}')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
    
    plt.tight_layout()
    plt.show()

    # 4. Fuerzas del modelo lineal - Longitudinal (X_o)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    xo_ruedas = [
        ('X_o_FL', 'Front Left (FL)', 'blue'),
        ('X_o_FR', 'Front Right (FR)', 'red'),
        ('X_o_RL', 'Rear Left (RL)', 'green'),
        ('X_o_RR', 'Rear Right (RR)', 'orange')
    ]
    
    for idx, (col, label, color) in enumerate(xo_ruedas):
        axes[idx].plot(df_track['time'], df_track[col], color=color, label=label, linewidth=1)
        axes[idx].set_xlabel('Tiempo [s]')
        axes[idx].set_ylabel('Fuerza Longitudinal [N]')
        axes[idx].set_title(f'Fuerza Longitudinal (X_o) en {label}')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
    
    plt.tight_layout()
    plt.show()

    # 5. Fuerzas del modelo lineal - Lateral (Y_o)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    yo_ruedas = [
        ('Y_o_FL', 'Front Left (FL)', 'blue'),
        ('Y_o_FR', 'Front Right (FR)', 'red'),
        ('Y_o_RL', 'Rear Left (RL)', 'green'),
        ('Y_o_RR', 'Rear Right (RR)', 'orange')
    ]
    
    for idx, (col, label, color) in enumerate(yo_ruedas):
        axes[idx].plot(df_track['time'], df_track[col], color=color, label=label, linewidth=1)
        axes[idx].set_xlabel('Tiempo [s]')
        axes[idx].set_ylabel('Fuerza Lateral [N]')
        axes[idx].set_title(f'Fuerza Lateral (Y_o) en {label}')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
    
    plt.tight_layout()
    plt.show()

def print_max_cargas_friccion(df_track, etapa):
    """
    Imprime los valores máximos de carga y fricción por rueda.
    """
    print(f"\n{'='*60}")
    print(f"MÁXIMOS POR RUEDA ({etapa}):")
    print(f"{'='*60}")
    
    print("\nCARGAS NORMALES MÁXIMAS:")
    for rueda, label in zip(
        ['Fz_FL', 'Fz_FR', 'Fz_RL', 'Fz_RR'],
        ['Front Left (FL)', 'Front Right (FR)', 'Rear Left (RL)', 'Rear Right (RR)']
    ):
        max_carga = df_track[rueda].max()
        print(f"  {label}: {max_carga:.2f} N")
    
    print("\nFUERZAS DE FRICCIÓN MÁXIMAS:")
    for rueda, label in zip(
        ['F_fric_FL', 'F_fric_FR', 'F_fric_RL', 'F_fric_RR'],
        ['Front Left (FL)', 'Front Right (FR)', 'Rear Left (RL)', 'Rear Right (RR)']
    ):
        max_fric = df_track[rueda].max()
        print(f"  {label}: {max_fric:.2f} N")
    
    print("\nFUERZAS LONGITUDINALES MÁXIMAS (Modelo Lineal):")
    for rueda, label in zip(
        ['X_o_FL', 'X_o_FR', 'X_o_RL', 'X_o_RR'],
        ['Front Left (FL)', 'Front Right (FR)', 'Rear Left (RL)', 'Rear Right (RR)']
    ):
        max_xo = df_track[rueda].max()
        print(f"  {label}: {max_xo:.2f} N")
    
    print("\nFUERZAS LATERALES MÁXIMAS (Modelo Lineal):")
    for rueda, label in zip(
        ['Y_o_FL', 'Y_o_FR', 'Y_o_RL', 'Y_o_RR'],
        ['Front Left (FL)', 'Front Right (FR)', 'Rear Left (RL)', 'Rear Right (RR)']
    ):
        max_yo = df_track[rueda].max()
        print(f"  {label}: {max_yo:.2f} N")

def analyze_critical_points(df_track):
    """
    Analiza puntos críticos de la carrera donde ocurren las fuerzas máximas.
    """
    print(f"\n{'='*60}")
    print("ANÁLISIS DE PUNTOS CRÍTICOS")
    print(f"{'='*60}")
    
    # Encontrar índice de máxima fuerza total combinada
    total_force = (df_track['F_fric_FL'] + df_track['F_fric_FR'] + 
                   df_track['F_fric_RL'] + df_track['F_fric_RR'])
    max_force_idx = total_force.idxmax()
    
    max_time = df_track.loc[max_force_idx, 'time']
    max_ax = df_track.loc[max_force_idx, 'ax_g']
    max_ay = df_track.loc[max_force_idx, 'ay_g']
    
    print(f"Punto de máxima fuerza total:")
    print(f"  Tiempo: {max_time:.2f} s")
    print(f"  Aceleración Lateral: {max_ax:.3f} g")
    print(f"  Aceleración Longitudinal: {max_ay:.3f} g")
    
    # Mostrar fuerzas en ese punto crítico
    print(f"\nFuerzas en el punto crítico:")
    for rueda in ['FL', 'FR', 'RL', 'RR']:
        fz = df_track.loc[max_force_idx, f'Fz_{rueda}']
        ffric = df_track.loc[max_force_idx, f'F_fric_{rueda}']
        xo = df_track.loc[max_force_idx, f'X_o_{rueda}']
        yo = df_track.loc[max_force_idx, f'Y_o_{rueda}']
        print(f"  {rueda}: Fz={fz:.1f}N, F_fric={ffric:.1f}N, X_o={xo:.1f}N, Y_o={yo:.1f}N")

# -------------------------------
# EJECUCIÓN PRINCIPAL
# -------------------------------
if __name__ == "__main__":
    print("ANÁLISIS DE NEUMÁTICOS DE COMPETICIÓN - VERSIÓN POTENCIADA")
    print("=" * 60)
    
    # --- INGRESA LA RUTA DEL ARCHIVO CSV A ANALIZAR ---
    csv_path = input("Ruta del archivo CSV a analizar: ").strip()

    # Cargar archivo CSV manualmente
    if not os.path.exists(csv_path):
        print(f"Archivo CSV no encontrado: {csv_path}")
        exit(1)

    df_raw = pd.read_csv(csv_path)
    time_col = 'loggingSample(N)'
    ax_col = 'accelerometerAccelerationX(G)'
    ay_col = 'accelerometerAccelerationY(G)'
    az_col = 'accelerometerAccelerationZ(G)'

    track_data = {
        'time': df_raw[time_col].tolist(),
        'ax_g': df_raw[ax_col].tolist(),
        'ay_g': df_raw[ay_col].tolist(),
        'az_g': df_raw[az_col].tolist(),
    }

    print(f"\nDatos cargados: {len(track_data['time'])} muestras")
    print(f"Parámetros de neumáticos:")
    print(f"  kK (rigidez longitudinal): {kK}")
    print(f"  k_alpha (rigidez lateral deriva): {k_alpha} 1/rad")
    print(f"  k_gamma (rigidez lateral camber): {k_gamma} 1/rad")

    # Analizar datos dinámicos
    print("\nAnalizando datos de pista...")
    df_track = analyze_track_data(track_data, mu, az_offset=-0.1)

    # --- Calcular máximos pares de aceleraciones ---
    max_idx = np.argmax(np.abs(df_track['ax_g']) + np.abs(df_track['ay_g']))
    max_ax = df_track.loc[max_idx, 'ax_g']
    max_ay = df_track.loc[max_idx, 'ay_g']
    max_time = df_track.loc[max_idx, 'time']

    print(f"\nMáximo par de aceleraciones:")
    print(f"  Tiempo: {max_time:.2f} s | ax_g: {max_ax:.3f} | ay_g: {max_ay:.3f}")

    print("\nPRIMERAS 5 MUESTRAS ANALIZADAS:")
    print(df_track.head().to_string(index=False))
    
    # Análisis completo
    print_max_cargas_friccion(df_track, "ANÁLISIS COMPLETO")
    analyze_critical_points(df_track)

    # Visualización de resultados
    print("\nGenerando gráficos...")
    visualize_results(df_track)
    
    print("\n¡Análisis completado! Puedes eliminar f_neumaticos.py y usar este archivo como principal.")