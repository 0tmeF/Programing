# -*- coding: utf-8 -*-
"""
Módulo analyze_track_data: funciones reutilizables para tire_dynamics_v16.py

Contiene:
 - tire_forces
 - calculate_tire_forces_linear
 - load_mu_vs_T_fit_from_summary
 - mu_from_T, invert_mu_to_T
 - analyze_track_data (principal, importable)
 - utilidades para cargar/limpiar CSV y visualizar resultados

Guardar en: /Users/carlos/Programming/PIM/analisis_dinamico/analyze_track_data.py
"""
import os
import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# PARÁMETROS CONSTANTES DEL VEHÍCULO (ajustables)
# -------------------------------
M = 1000.0         # Masa del vehículo [kg]
g = 9.81           # Gravedad [m/s²]
a = 0.92           # Distancia CG al eje delantero [m]
b = 1.53           # Distancia CG al eje trasero [m]
h_cg = 0.45        # Altura CG [m]
track_front = 1.70 # Trocha delantera [m]
track_rear = 1.70  # Trocha trasera [m]
mu_default = 1.0   # Coeficiente de fricción por defecto

# -------------------------------
# PARÁMETROS DE LOS NEUMÁTICOS (modelo lineal)
# -------------------------------
kK = 20.0
k_alpha = 10.0
k_gamma = 1.0

# -------------------------------
# CÁLCULO DE CARGAS Y FRICCIÓN
# -------------------------------
def tire_forces(ax_g: float, ay_g: float, az_g: float, mu: float = mu_default) -> tuple:
    """
    Calcula cargas normales por rueda y fuerzas de fricción simples.
    ax_g, ay_g, az_g se entregan en g (no m/s2). Se retorna (Fz_FL, Fz_FR, Fz_RL, Fz_RR, friction_dict).
    """
    ax = float(ax_g) * g
    ay = float(ay_g) * g
    az = float(az_g) * g
    w = a + b

    # Cargas estáticas por eje
    FzF0 = M * g * b / w
    FzR0 = M * g * a / w

    # Transferencia longitudinal por aceleración ax
    dF_long = M * ax * h_cg / w
    FzF = FzF0 - dF_long
    FzR = FzR0 + dF_long

    # Transferencia lateral aproximada (proporcional)
    dF_lat_front = (M * ay * h_cg) * (b / w) / max(0.0001, track_front)
    dF_lat_rear = (M * ay * h_cg) * (a / w) / max(0.0001, track_rear)

    # Distribución por rueda (no-negativa)
    Fz_FL = max(0.0, FzF / 2.0 - dF_lat_front / 2.0)
    Fz_FR = max(0.0, FzF / 2.0 + dF_lat_front / 2.0)
    Fz_RL = max(0.0, FzR / 2.0 - dF_lat_rear / 2.0)
    Fz_RR = max(0.0, FzR / 2.0 + dF_lat_rear / 2.0)

    friction_forces = {
        'FL': mu * Fz_FL,
        'FR': mu * Fz_FR,
        'RL': mu * Fz_RL,
        'RR': mu * Fz_RR
    }

    return Fz_FL, Fz_FR, Fz_RL, Fz_RR, friction_forces

# -------------------------------
# MODELO LINEAL SIMPLE POR RUEDA
# -------------------------------
def calculate_tire_forces_linear(Fz_FL: float, Fz_FR: float, Fz_RL: float, Fz_RR: float,
                                 K: Tuple[float,float,float,float]=None,
                                 alpha: Tuple[float,float,float,float]=None,
                                 gamma: Tuple[float,float,float,float]=None) -> pd.DataFrame:
    """
    Modelo lineal simplificado para X_o, Y_o por rueda.
    """
    if K is None:
        K = (-0.01, -0.01, -0.01, -0.01)
    if alpha is None:
        alpha = (0.0, 0.0, 0.0, 0.0)
    if gamma is None:
        gamma = (np.deg2rad(-0.5), np.deg2rad(-0.5), np.deg2rad(-0.5), np.deg2rad(-0.5))

    ruedas = ['FL','FR','RL','RR']
    Fz_values = [Fz_FL, Fz_FR, Fz_RL, Fz_RR]
    rows = []
    for i, r in enumerate(ruedas):
        Nw = float(Fz_values[i])
        Xo = kK * Nw * float(K[i])
        Yo = k_alpha * Nw * float(alpha[i]) + k_gamma * Nw * float(gamma[i])
        rows.append({
            'Rueda': r,
            'N [N]': Nw,
            'K [-]': K[i],
            'alpha [rad]': alpha[i],
            'gamma [rad]': gamma[i],
            'X_o [N]': Xo,
            'Y_o [N]': Yo,
            'Fuerza_Total [N]': float(np.hypot(Xo, Yo))
        })
    return pd.DataFrame(rows)

# -------------------------------
# HELPERS: CARGAR AJUSTE μ(T)
# -------------------------------
def load_mu_vs_T_fit_from_summary(resumen_csv_path: str) -> Optional[List[float]]:
    """
    Lee 'fit_coeffs_mu_T' o como fallback 'fit_poly_str' en el CSV resumen y devuelve lista de coeficientes
    en orden high->low (compat. numpy.polyval). Retorna None si no hay ajuste.
    """
    if not os.path.exists(resumen_csv_path):
        raise FileNotFoundError(f"No existe resumen CSV: {resumen_csv_path}")
    df = pd.read_csv(resumen_csv_path)
    # 1) intentar cargar coeficientes JSON directamente
    if 'fit_coeffs_mu_T' in df.columns:
        vals = df['fit_coeffs_mu_T'].dropna().values
        if len(vals) > 0:
            raw = vals[-1]
            try:
                coeffs = json.loads(raw) if isinstance(raw, str) else raw
                return [float(c) for c in coeffs]
            except Exception:
                pass
    # 2) fallback: intentar usar la cadena polinomial legible ('fit_poly_str') si existe
    if 'fit_poly_str' in df.columns:
        vals = df['fit_poly_str'].dropna().values
        if len(vals) > 0:
            poly_str = vals[-1]
            try:
                # detectar grado aproximado buscando "T**n"
                import re
                powers = re.findall(r'T\*\*(\d+)', poly_str)
                deg = int(max(powers, default=1))
                deg = max(1, min(8, deg))  # limitar grado razonable
                # construir función evaluable: eval en entorno controlado
                def fT(t):
                    # permitir sólo la variable T y operaciones aritméticas
                    return float(eval(poly_str, {"__builtins__": None}, {"T": float(t)}))
                # samplear y ajustar con polyfit
                xs = np.linspace(20.0, 140.0, 121)  # rango típico
                ys = np.array([fT(x) for x in xs], dtype=float)
                p = np.polyfit(xs, ys, deg)
                return [float(co) for co in p]
            except Exception:
                pass
    return None

def mu_from_T(T: float, coeffs: List[float]) -> float:
    """Evalúa μ(T) para un T dado con coeficientes (high->low)."""
    return float(np.polyval(coeffs, float(T)))

def invert_mu_to_T(mu_target: float, coeffs: List[float], temp_bounds: Tuple[float,float]=(0.0,200.0)) -> Optional[float]:
    """
    Resuelve μ(T) = mu_target. Devuelve raíz real en temp_bounds más cercana al centro.
    Si no hay solución válida, devuelve None y no lanza excepción.
    """
    if coeffs is None or len(coeffs) < 2:
        return None
    p = np.array(coeffs, dtype=float).copy()
    p[-1] = p[-1] - float(mu_target)
    try:
        roots = np.roots(p)
    except Exception:
        return None
    real_roots = [r.real for r in roots if np.isreal(r)]
    valid = [r for r in real_roots if temp_bounds[0] <= r <= temp_bounds[1]]
    if not valid:
        # no validar raise; informar al caller
        return None
    center = 0.5 * (temp_bounds[0] + temp_bounds[1])
    selected = min(valid, key=lambda x: abs(x - center))
    return float(selected)

# -------------------------------
# FUNCION PRINCIPAL: ANALIZAR DATOS DE PISTA
# -------------------------------
def analyze_track_data(track_data: dict, mu: float = mu_default,
                       ax_offset: float = -0.080, ay_offset: float = -0.090, az_offset: float = -0.1,
                       mu_vs_T_coeffs: Optional[List[float]] = None,
                       temp_bounds: Tuple[float,float] = (20.0, 140.0)) -> pd.DataFrame:
    """
    Analiza track_data (diccionario con listas 'time','ax_g','ay_g','az_g').
    Si se entrega mu_vs_T_coeffs, para cada muestra se estima mu_est desde el modelo lineal
    y se invierte μ(T) obteniendo predicted_temp_C.
    """
    times = track_data.get('time', [])
    n = len(times)
    rows = []
    for i in range(n):
        try:
            ax_g = float(track_data['ax_g'][i]) + ax_offset
            ay_g = float(track_data['ay_g'][i]) + ay_offset
            # usar convención + az_offset (consistente)
            az_g = float(track_data['az_g'][i]) + az_offset
        except Exception:
            # dato inválido -> omitir
            continue

        if not (np.isfinite(ax_g) and np.isfinite(ay_g) and np.isfinite(az_g)):
            continue

        Fz_FL, Fz_FR, Fz_RL, Fz_RR, friction = tire_forces(ax_g, ay_g, az_g, mu)

        tdf = calculate_tire_forces_linear(Fz_FL, Fz_FR, Fz_RL, Fz_RR)

        Xs = []
        Ys = []
        Ns = []
        for wheel in ['FL','FR','RL','RR']:
            roww = tdf[tdf['Rueda'] == wheel].iloc[0]
            Xs.append(float(roww['X_o [N]']))
            Ys.append(float(roww['Y_o [N]']))
            Ns.append(max(1e-9, float(roww['N [N]'])))

        forces_mag = [float(np.hypot(x,y)) for x,y in zip(Xs,Ys)]
        total_force = sum(forces_mag)
        total_N = sum(Ns)
        mu_est = total_force / total_N if total_N > 0 else 0.0

        predicted_temp = None
        if mu_vs_T_coeffs is not None:
            predicted_temp = invert_mu_to_T(mu_est, mu_vs_T_coeffs, temp_bounds=temp_bounds)
            # informar si no se pudo invertir
            if predicted_temp is None:
                # no raise: caller decide qué hacer
                pass

        rows.append({
            'time': times[i],
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
            'X_o_FL': tdf.loc[tdf['Rueda']=='FL','X_o [N]'].iloc[0],
            'Y_o_FL': tdf.loc[tdf['Rueda']=='FL','Y_o [N]'].iloc[0],
            'X_o_FR': tdf.loc[tdf['Rueda']=='FR','X_o [N]'].iloc[0],
            'Y_o_FR': tdf.loc[tdf['Rueda']=='FR','Y_o [N]'].iloc[0],
            'X_o_RL': tdf.loc[tdf['Rueda']=='RL','X_o [N]'].iloc[0],
            'Y_o_RL': tdf.loc[tdf['Rueda']=='RL','Y_o [N]'].iloc[0],
            'X_o_RR': tdf.loc[tdf['Rueda']=='RR','X_o [N]'].iloc[0],
            'Y_o_RR': tdf.loc[tdf['Rueda']=='RR','Y_o [N]'].iloc[0],
            'mu_est': round(mu_est, 4),
            'predicted_temp_C': round(predicted_temp, 2) if predicted_temp is not None else None
        })

    return pd.DataFrame(rows)

# -------------------------------
# UTILIDADES (puedes importarlas)
# -------------------------------
def find_data_start(csv_path: str, ax_col='accelerometerAccelerationX(G)', ay_col='accelerometerAccelerationY(G)',
                    threshold: float = 0.2, window: int = 2000) -> Optional[int]:
    """
    Busca primer índice con bloque útil de datos según aceleraciones.
    """
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if ax_col not in df.columns or ay_col not in df.columns:
        return None
    abs_acc = np.abs(df[ax_col]) + np.abs(df[ay_col])
    for i in range(0, max(1, len(df) - window)):
        if np.all(abs_acc.iloc[i:i+window] > threshold):
            return i
    return None

def load_and_clean_csv(csv_path: str, output_name: str, ax_col='accelerometerAccelerationX(G)',
                       ay_col='accelerometerAccelerationY(G)', az_col='accelerometerAccelerationZ(G)',
                       time_col='loggingSample(N)', threshold=0.3, window=10, min_time=None):
    """
    Carga y limpia el CSV; guarda archivo limpio y devuelve track_data dict.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df_raw = pd.read_csv(csv_path)
    if time_col not in df_raw.columns:
        raise KeyError(f"time_col '{time_col}' no encontrado en CSV")
    if min_time is not None:
        df_raw = df_raw[df_raw[time_col] >= min_time].reset_index(drop=True)
    else:
        start_idx = find_data_start(csv_path, ax_col, ay_col, threshold=threshold, window=window)
        if start_idx is not None:
            df_raw = df_raw.iloc[start_idx:].reset_index(drop=True)
    df_raw.to_csv(output_name, index=False)
    track_data = {
        'time': df_raw[time_col].tolist(),
        'ax_g': df_raw[ax_col].tolist(),
        'ay_g': df_raw[ay_col].tolist(),
        'az_g': df_raw[az_col].tolist(),
    }
    return track_data

def visualize_results(df_track: pd.DataFrame) -> None:
    """Gráficos básicos (importable)."""
    if df_track.empty:
        print("No hay datos para mostrar.")
        return
    plt.figure(figsize=(10,4))
    plt.plot(df_track['time'], df_track['ax_g'], label='ax_g')
    plt.plot(df_track['time'], df_track['ay_g'], label='ay_g')
    plt.xlabel('Tiempo [s]'); plt.ylabel('Aceleración [g]')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# -------------------------------
# EXPORTS
# -------------------------------
__all__ = [
    'tire_forces', 'calculate_tire_forces_linear',
    'load_mu_vs_T_fit_from_summary', 'mu_from_T', 'invert_mu_to_T',
    'analyze_track_data', 'find_data_start', 'load_and_clean_csv', 'visualize_results'
]

# Módulo listo para importar desde tire_dynamics_v16.py