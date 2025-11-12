# -*- coding: utf-8 -*-
"""
Análisis térmico y de fuerzas en neumáticos - archivo completo listo para guardar.
Incluye:
 - funciones de cálculo de cargas y modelo lineal por rueda
 - helpers para cargar ajuste μ(T) producido por BancoEnsayos
 - analyze_track_data que devuelve un DataFrame con mu_est y predicted_temp_C
 - funciones de visualización e informes básicos
Autor: generar por GitHub Copilot (adaptar parámetros según necesidad)
"""
import os
import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
mu_default = 1.0   # Coeficiente de fricción por defecto (solo para cálculo de fricción máxima)

# -------------------------------
# PARÁMETROS DE LOS NEUMÁTICOS (modelo lineal)
# -------------------------------
kK = 20.0          # Rigidez longitudinal (adimensional)
k_alpha = 10.0     # Rigidez lateral por ángulo de deriva [N/rad per N] (adimensional*load)
k_gamma = 1.0      # Rigidez por camber [N/rad per N] (adimensional*load)

# -------------------------------
# FUNCIONES BÁSICAS DE NEUMÁTICOS Y MODELO LINEAL
# -------------------------------

def tire_forces(ax_g: float, ay_g: float, az_g: float, mu: float = mu_default) -> tuple:
    """
    Calcula cargas normales por rueda y fuerzas de fricción estáticas simples.
    ax_g, ay_g, az_g: aceleraciones en g
    Retorna (Fz_FL, Fz_FR, Fz_RL, Fz_RR, friction_dict)
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

def calculate_tire_forces_linear(Fz_FL: float, Fz_FR: float, Fz_RL: float, Fz_RR: float,
                                 K: Tuple[float,float,float,float]=None,
                                 alpha: Tuple[float,float,float,float]=None,
                                 gamma: Tuple[float,float,float,float]=None) -> pd.DataFrame:
    """
    Modelo lineal simplificado para X_o, Y_o por rueda.
    K: slip ratios (default -0.01)
    alpha: slip angles [rad] (default 0)
    gamma: camber [rad] (default -0.5 deg)
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
# HELPERS: CARGAR AJUSTE μ(T) DESDE BANCOENSAYOS
# -------------------------------

def load_mu_vs_T_fit_from_summary(resumen_csv_path: str) -> Optional[List[float]]:
    """
    Lee la columna 'fit_coeffs_mu_T' en el CSV resumen y devuelve la lista de coeficientes
    en el orden usado por numpy.polyval (coef listo high->low).
    """
    if not os.path.exists(resumen_csv_path):
        raise FileNotFoundError(f"No existe resumen CSV: {resumen_csv_path}")
    df = pd.read_csv(resumen_csv_path)
    if 'fit_coeffs_mu_T' not in df.columns:
        return None
    vals = df['fit_coeffs_mu_T'].dropna().values
    if len(vals) == 0:
        return None
    # tomar primer no-nulo
    raw = vals[0]
    try:
        coeffs = json.loads(raw) if isinstance(raw, str) else raw
        return [float(c) for c in coeffs]
    except Exception:
        return None

def mu_from_T(T: float, coeffs: List[float]) -> float:
    """Evalúa μ(T) con coeficientes (high->low)."""
    return float(np.polyval(coeffs, float(T)))

def invert_mu_to_T(mu_target: float, coeffs: List[float], temp_bounds: Tuple[float,float]=(0.0,200.0)) -> Optional[float]:
    """
    Resuelve p(T) = mu_target. Devuelve la raíz real dentro de temp_bounds más cercana al centro.
    """
    p = np.array(coeffs, dtype=float).copy()
    p[-1] = p[-1] - float(mu_target)
    roots = np.roots(p)
    real_roots = [r.real for r in roots if np.isreal(r)]
    valid = [r for r in real_roots if temp_bounds[0] <= r <= temp_bounds[1]]
    if not valid:
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
    Devuelve DataFrame con columnas relevantes.
    """
    n = len(track_data.get('time', []))
    rows = []
    for i in range(n):
        ax_g = float(track_data['ax_g'][i]) + ax_offset
        ay_g = float(track_data['ay_g'][i]) + ay_offset
        az_g = float(track_data['az_g'][i]) - az_offset

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

        rows.append({
            'time': track_data['time'][i],
            'ax_g': ax_g,
            'ay_g': ay_g,
            'az_g': az_g,
            'Fz_FL': round(Fz_FL,2),
            'Fz_FR': round(Fz_FR,2),
            'Fz_RL': round(Fz_RL,2),
            'Fz_RR': round(Fz_RR,2),
            'F_fric_FL': round(friction['FL'],2),
            'F_fric_FR': round(friction['FR'],2),
            'F_fric_RL': round(friction['RL'],2),
            'F_fric_RR': round(friction['RR'],2),
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

    df = pd.DataFrame(rows)
    # añadir columnas de resumen por rueda (Fz máximas, fricción máximas)
    return df

# -------------------------------
# UTILIDADES: VISUALIZACION E INFORMES
# -------------------------------

def visualize_results(df_track: pd.DataFrame, params: dict = None) -> None:
    """Genera gráficos básicos (ax, ay, cargas y fuerzas X/Y por rueda)."""
    if params is None:
        params = {}
    if df_track.empty:
        print("No hay datos para mostrar.")
        return

    plt.figure(figsize=(10,4))
    plt.plot(df_track['time'], df_track['ax_g'], label='ax_g')
    plt.plot(df_track['time'], df_track['ay_g'], label='ay_g')
    plt.xlabel('Tiempo [s]'); plt.ylabel('Aceleración [g]')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    ruedas = [('Fz_FL','FL'),('Fz_FR','FR'),('Fz_RL','RL'),('Fz_RR','RR')]
    fig, axs = plt.subplots(2,2,figsize=(10,6))
    axs = axs.flatten()
    for i,(col,lab) in enumerate(ruedas):
        axs[i].plot(df_track['time'], df_track[col])
        axs[i].set_title(f'Carga normal {lab}'); axs[i].grid(alpha=0.3)
    plt.tight_layout(); plt.show()

def print_max_cargas_friccion(df_track: pd.DataFrame, etapa: str = "RESULTADO"):
    print(f"\n--- MÁXIMOS ({etapa}) ---")
    for col in ['Fz_FL','Fz_FR','Fz_RL','Fz_RR']:
        if col in df_track.columns:
            print(f"{col}: {df_track[col].max():.1f} N")
    for col in ['F_fric_FL','F_fric_FR','F_fric_RL','F_fric_RR']:
        if col in df_track.columns:
            print(f"{col}: {df_track[col].max():.1f} N")

def analyze_critical_points(df_track: pd.DataFrame):
    if df_track.empty:
        print("No hay datos.")
        return
    total_force = df_track[['F_fric_FL','F_fric_FR','F_fric_RL','F_fric_RR']].sum(axis=1)
    idx = total_force.idxmax()
    print(f"Punto crítico en t={df_track.loc[idx,'time']}: mu_est={df_track.loc[idx,'mu_est']} predicted_temp={df_track.loc[idx,'predicted_temp_C']}")

# -------------------------------
# EJECUCIÓN PRINCIPAL (CLI)
# -------------------------------
if __name__ == "__main__":
    print("TIRE DYNAMICS - análisis integrado con μ(T) fit")
    csv_path = input("Ruta del archivo CSV a analizar: ").strip()
    if not os.path.exists(csv_path):
        print("CSV no encontrado."); raise SystemExit(1)

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

    # intentar cargar ajuste μ(T) - modifica la ruta según tu output de BancoEnsayos
    resumen_csv = os.path.join(os.path.dirname(__file__), '..', 'banco_ensayos', 'outputs', 'resumen_muef_por_hoja.csv')
    resumen_csv = os.path.abspath(resumen_csv)
    muT_coeffs = None
    try:
        if os.path.exists(resumen_csv):
            muT_coeffs = load_mu_vs_T_fit_from_summary(resumen_csv)
            print(f"μ(T) fit cargado desde: {resumen_csv}")
        else:
            print(f"No se encontró resumen μ(T) en {resumen_csv}; se usará μ constante.")
    except Exception as e:
        print(f"Warning cargando μ(T): {e}")

    df_track = analyze_track_data(track_data, mu=mu_default, az_offset=-0.1, mu_vs_T_coeffs=muT_coeffs)

    print("\nPRIMERAS 5 MUESTRAS:")
    print(df_track.head().to_string(index=False))

    print_max_cargas_friccion(df_track, "ANÁLISIS COMPLETO")
    analyze_critical_points(df_track)
    visualize_results(df_track)

    print("\nFin del análisis.")
```# filepath: /Users/carlos/Programming/PIM/analisis_dinamico/tire_dynamics_v16.py
# -*- coding: utf-8 -*-
"""
Análisis térmico y de fuerzas en neumáticos - archivo completo listo para guardar.
Incluye:
 - funciones de cálculo de cargas y modelo lineal por rueda
 - helpers para cargar ajuste μ(T) producido por BancoEnsayos
 - analyze_track_data que devuelve un DataFrame con mu_est y predicted_temp_C
 - funciones de visualización e informes básicos
Autor: generar por GitHub Copilot (adaptar parámetros según necesidad)
"""
import os
import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
mu_default = 1.0   # Coeficiente de fricción por defecto (solo para cálculo de fricción máxima)

# -------------------------------
# PARÁMETROS DE LOS NEUMÁTICOS (modelo lineal)
# -------------------------------
kK = 20.0          # Rigidez longitudinal (adimensional)
k_alpha = 10.0     # Rigidez lateral por ángulo de deriva [N/rad per N] (adimensional*load)
k_gamma = 1.0      # Rigidez por camber [N/rad per N] (adimensional*load)

# -------------------------------
# FUNCIONES BÁSICAS DE NEUMÁTICOS Y MODELO LINEAL
# -------------------------------

def tire_forces(ax_g: float, ay_g: float, az_g: float, mu: float = mu_default) -> tuple:
    """
    Calcula cargas normales por rueda y fuerzas de fricción estáticas simples.
    ax_g, ay_g, az_g: aceleraciones en g
    Retorna (Fz_FL, Fz_FR, Fz_RL, Fz_RR, friction_dict)
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

def calculate_tire_forces_linear(Fz_FL: float, Fz_FR: float, Fz_RL: float, Fz_RR: float,
                                 K: Tuple[float,float,float,float]=None,
                                 alpha: Tuple[float,float,float,float]=None,
                                 gamma: Tuple[float,float,float,float]=None) -> pd.DataFrame:
    """
    Modelo lineal simplificado para X_o, Y_o por rueda.
    K: slip ratios (default -0.01)
    alpha: slip angles [rad] (default 0)
    gamma: camber [rad] (default -0.5 deg)
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
# HELPERS: CARGAR AJUSTE μ(T) DESDE BANCOENSAYOS
# -------------------------------

def load_mu_vs_T_fit_from_summary(resumen_csv_path: str) -> Optional[List[float]]:
    """
    Lee la columna 'fit_coeffs_mu_T' en el CSV resumen y devuelve la lista de coeficientes
    en el orden usado por numpy.polyval (coef listo high->low).
    """
    if not os.path.exists(resumen_csv_path):
        raise FileNotFoundError(f"No existe CSV: {resumen_csv_path}")
    df = pd.read_csv(resumen_csv_path)
    if 'fit_coeffs_mu_T' not in df.columns:
        return None
    vals = df['fit_coeffs_mu_T'].dropna().values
    if len(vals) == 0:
        return None
    # tomar primer no-nulo
    raw = vals[0]
    try:
        coeffs = json.loads(raw) if isinstance(raw, str) else raw
        return [float(c) for c in coeffs]
    except Exception:
        return None

def mu_from_T(T: float, coeffs: List[float]) -> float:
    """Evalúa μ(T) con coeficientes (high->low)."""
    return float(np.polyval(coeffs, float(T)))

def invert_mu_to_T(mu_target: float, coeffs: List[float], temp_bounds: Tuple[float,float]=(0.0,200.0)) -> Optional[float]:
    """
    Resuelve p(T) = mu_target. Devuelve la raíz real dentro de temp_bounds más cercana al centro.
    """
    p = np.array(coeffs, dtype=float).copy()
    p[-1] = p[-1] - float(mu_target)
    roots = np.roots(p)
    real_roots = [r.real for r in roots if np.isreal(r)]
    valid = [r for r in real_roots if temp_bounds[0] <= r <= temp_bounds[1]]
    if not valid:
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
    Devuelve DataFrame con columnas relevantes.
    """
    n = len(track_data.get('time', []))
    rows = []
    for i in range(n):
        ax_g = float(track_data['ax_g'][i]) + ax_offset
        ay_g = float(track_data['ay_g'][i]) + ay_offset
        az_g = float(track_data['az_g'][i]) - az_offset

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

        rows.append({
            'time': track_data['time'][i],
            'ax_g': ax_g,
            'ay_g': ay_g,
            'az_g': az_g,
            'Fz_FL': round(Fz_FL,2),
            'Fz_FR': round(Fz_FR,2),
            'Fz_RL': round(Fz_RL,2),
            'Fz_RR': round(Fz_RR,2),
            'F_fric_FL': round(friction['FL'],2),
            'F_fric_FR': round(friction['FR'],2),
            'F_fric_RL': round(friction['RL'],2),
            'F_fric_RR': round(friction['RR'],2),
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

    df = pd.DataFrame(rows)
    # añadir columnas de resumen por rueda (Fz máximas, fricción máximas)
    return df

# -------------------------------
# UTILIDADES: VISUALIZACION E INFORMES
# -------------------------------

def visualize_results(df_track: pd.DataFrame, params: dict = None) -> None:
    """Genera gráficos básicos (ax, ay, cargas y fuerzas X/Y por rueda)."""
    if params is None:
        params = {}
    if df_track.empty:
        print("No hay datos para mostrar.")
        return

    plt.figure(figsize=(10,4))
    plt.plot(df_track['time'], df_track['ax_g'], label='ax_g')
    plt.plot(df_track['time'], df_track['ay_g'], label='ay_g')
    plt.xlabel('Tiempo [s]'); plt.ylabel('Aceleración [g]')
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

    ruedas = [('Fz_FL','FL'),('Fz_FR','FR'),('Fz_RL','RL'),('Fz_RR','RR')]
    fig, axs = plt.subplots(2,2,figsize=(10,6))
    axs = axs.flatten()
    for i,(col,lab) in enumerate(ruedas):
        axs[i].plot(df_track['time'], df_track[col])
        axs[i].set_title(f'Carga normal {lab}'); axs[i].grid(alpha=0.3)
    plt.tight_layout(); plt.show()

def print_max_cargas_friccion(df_track: pd.DataFrame, etapa: str = "RESULTADO"):
    print(f"\n--- MÁXIMOS ({etapa}) ---")
    for col in ['Fz_FL','Fz_FR','Fz_RL','Fz_RR']:
        if col in df_track.columns:
            print(f"{col}: {df_track[col].max():.1f} N")
    for col in ['F_fric_FL','F_fric_FR','F_fric_RL','F_fric_RR']:
        if col in df_track.columns:
            print(f"{col}: {df_track[col].max():.1f} N")

def analyze_critical_points(df_track: pd.DataFrame):
    if df_track.empty:
        print("No hay datos.")
        return
    total_force = df_track[['F_fric_FL','F_fric_FR','F_fric_RL','F_fric_RR']].sum(axis=1)
    idx = total_force.idxmax()
    print(f"Punto crítico en t={df_track.loc[idx,'time']}: mu_est={df_track.loc[idx,'mu_est']} predicted_temp={df_track.loc[idx,'predicted_temp_C']}")

# -------------------------------
# EJECUCIÓN PRINCIPAL (CLI)
# -------------------------------
if __name__ == "__main__":
    print("TIRE DYNAMICS - análisis integrado con μ(T) fit")
    csv_path = input("Ruta del archivo CSV a analizar: ").strip()
    if not os.path.exists(csv_path):
        print("CSV no encontrado."); raise SystemExit(1)

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

    # intentar cargar ajuste μ(T) - modifica la ruta según tu output de BancoEnsayos
    resumen_csv = os.path.join(os.path.dirname(__file__), '..', 'banco_ensayos', 'outputs', 'resumen_muef_por_hoja.csv')
    resumen_csv = os.path.abspath(resumen_csv)
    muT_coeffs = None
    try:
        if os.path.exists(resumen_csv):
            muT_coeffs = load_mu_vs_T_fit_from_summary(resumen_csv)
            print(f"μ(T) fit cargado desde: {resumen_csv}")
        else:
            print(f"No se encontró resumen μ(T) en {resumen_csv}; se usará μ constante.")
    except Exception as e:
        print(f"Warning cargando μ(T): {e}")

    df_track = analyze_track_data(track_data, mu=mu_default, az_offset=-0.1, mu_vs_T_coeffs=muT_coeffs)

    print("\nPRIMERAS 5 MUESTRAS:")
    print(df_track.head().to_string(index=False))

    print_max_cargas_friccion(df_track, "ANÁLISIS COMPLETO")
    analyze_critical_points(df_track)
    visualize_results(df_track)

    print("\nFin del análisis.")