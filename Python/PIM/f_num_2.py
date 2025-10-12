# -*- coding: utf-8 -*-
"""
Análisis térmico de neumáticos de competición
---------------------------------------------
Este script modela la temperatura ideal de trabajo de neumáticos de competición a partir de datos experimentales.

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

def analyze_track_data(track_data: dict, mu: float, ax_offset: float = -0.080, ay_offset: float = -0.090, az_offset: float = -0.1,) -> pd.DataFrame:
    """
    Analiza los datos de pista y calcula las fuerzas por neumático.
    """
    results = []
    for i in range(len(track_data['time'])):
        ax_g = track_data['ax_g'][i]
        ay_g = track_data['ay_g'][i]
        # Corrige el offset de gravedad del sensor
        az_g = track_data['az_g'][i] - az_offset
        Fz_FL, Fz_FR, Fz_RL, Fz_RR, friction = tire_forces(ax_g, ay_g, az_g, mu)
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
    track_data = {
        'time': df_raw[time_col].tolist(),
        'ax_g': df_raw[ax_col].tolist(),
        'ay_g': df_raw[ay_col].tolist(),
    }
    print(f"Datos cargados: {len(track_data['time'])} muestras")
    return track_data

def visualize_results(df_track: pd.DataFrame, params: dict) -> None:
    """
    Genera gráficos en figuras separadas para el informe del equipo.
    """

    # Aceleración lateral (X)
    plt.figure(figsize=(7, 5))
    plt.plot(df_track['time'], df_track['ax_g'], 'b-', label='Aceleración Lateral (g)')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Aceleración Lateral [g]')
    plt.title('Aceleración Lateral en pista')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Aceleración longitudinal (Y)
    plt.figure(figsize=(7, 5))
    plt.plot(df_track['time'], df_track['ay_g'], 'r-', label='Aceleración Longitudinal (g)')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Aceleración Longitudinal [g]')
    plt.title('Aceleración Longitudinal en pista')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Carga normal - individual por rueda
    ruedas = [
        ('Fz_FL', 'Front Left (FL)', 'blue'),
        ('Fz_FR', 'Front Right (FR)', 'red'),
        ('Fz_RL', 'Rear Left (RL)', 'cyan'),
        ('Fz_RR', 'Rear Right (RR)', 'orange')
    ]
    for col, label, color in ruedas:
        plt.figure(figsize=(8, 4))
        plt.plot(df_track['time'], df_track[col], color=color, label=label)
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Carga normal [N]')
        plt.title(f'Carga normal en {label}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Fuerza de fricción - individual por rueda
    ruedas_fric = [
        ('F_fric_FL', 'Front Left (FL)', 'blue'),
        ('F_fric_FR', 'Front Right (FR)', 'red'),
        ('F_fric_RL', 'Rear Left (RL)', 'cyan'),
        ('F_fric_RR', 'Rear Right (RR)', 'orange')
    ]
    for col, label, color in ruedas_fric:
        plt.figure(figsize=(8, 4))
        plt.plot(df_track['time'], df_track[col], color=color, label=label)
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Fuerza de fricción [N]')
        plt.title(f'Fuerza de fricción en {label}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

def print_max_cargas_friccion(df_track, etapa):
    print(f"\nMÁXIMOS POR RUEDA ({etapa}):")
    for rueda, label in zip(
        ['Fz_FL', 'Fz_FR', 'Fz_RL', 'Fz_RR'],
        ['Front Left (FL)', 'Front Right (FR)', 'Rear Left (RL)', 'Rear Right (RR)']
    ):
        max_carga = df_track[rueda].max()
        print(f"  {label}: {max_carga:.2f} N (Carga máxima)")
    for rueda, label in zip(
        ['F_fric_FL', 'F_fric_FR', 'F_fric_RL', 'F_fric_RR'],
        ['Front Left (FL)', 'Front Right (FR)', 'Rear Left (RL)', 'Rear Right (RR)']
    ):
        max_fric = df_track[rueda].max()
        print(f"  {label}: {max_fric:.2f} N (Fricción máxima)")

# -------------------------------
# EJECUCIÓN PRINCIPAL
# -------------------------------
if __name__ == "__main__":
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
        'az_g': df_raw[az_col].tolist(),  # <-- AGREGAR ESTA LÍNEA
    }

    # Analizar datos dinámicos
    df_track = analyze_track_data(track_data, mu, az_offset=-0.1)

    # --- Calcular máximos pares de aceleraciones ---
    max_idx = np.argmax(np.abs(df_track['ax_g']) + np.abs(df_track['ay_g']))
    max_ax = df_track.loc[max_idx, 'ax_g']
    max_ay = df_track.loc[max_idx, 'ay_g']
    max_time = df_track.loc[max_idx, 'time']

    print(f"\nMáximo par de aceleraciones:")
    print(f"Tiempo: {max_time:.2f} s | ax_g: {max_ax:.3f} | ay_g: {max_ay:.3f}")

    print("\n4. TABLA DE RESULTADOS (primeras 5 filas)")
    print(df_track.head().to_string(index=False))
    print_max_cargas_friccion(df_track, "MANGA ANALIZADA")

    # Visualización de resultados
    visualize_results(df_track, params={})