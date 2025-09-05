#-- coding: utf-8 --
"""
ANÁLISIS COMPLETO NISSAN V16 - VERSIÓN 3.0
Simulación termodinámica completa con deriva lateral y optimización
Equipo: [Nombre del equipo]  
Autor: Carlos Caamaño
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------------------------------
# PARÁMETROS ESPECÍFICOS NISSAN V16
# -------------------------------
m = 1000.0    # masa [kg] - Nissan V16 con conductor y combustible
g = 9.81      # gravedad [m/s²]
a = 0.92      # distancia CG -> eje delantero [m] - Configuración V16
b = 1.53      # distancia CG -> eje trasero [m] - Tracción delantera
h_cg = 0.45   # altura del centro de masa [m] - Centro de gravedad bajo
tf = 1.70     # trocha delantera [m] - Ancho de vía V16
tr = 1.70     # trocha trasera [m]

# Parámetros neumáticos de competición (Semi-Slick)
k_K = 19.0       # rigidez longitudinal [18-20] - Competición agresiva
k_alfa = 16.0    # rigidez lateral por deriva [15-18] - Alta rigidez
k_gamma = 1.1    # rigidez lateral por camber [1.0-1.2]
μ_max_base = 1.45 # coeficiente máximo base a 85°C - Semi-slick de competición

# Parámetros de suspensión V16
relacion_subvirador = 0.65  # 65% subvirador - Típico tracción delantera
gamma_max = 2.5 * np.pi/180  # Camber máximo [rad] - Configuración agresiva

# -------------------------------
# SIMULACIÓN DE DATOS DE CARRERA V16
# -------------------------------
def simular_datos_carrera_v16():
    """Simula datos realistas de una vuelta de circuito para Nissan V16"""
    
    tiempo = np.arange(0, 90, 0.5)  # 90 segundos, datos cada 0.5s
    n_points = len(tiempo)
    
    # Simular perfil de velocidad típico de circuito
    velocidad = np.zeros(n_points)
    for i in range(n_points):
        if i < 20:  # Aceleración inicial
            velocidad[i] = 30 + i * 3
        elif i < 40:  # Recta rápida
            velocidad[i] = 90 + (i-20) * 1.5
        elif i < 50:  # Frenada para curva
            velocidad[i] = 150 - (i-40) * 6
        elif i < 65:  # Curva lenta
            velocidad[i] = 60 + (i-50) * 2
        elif i < 75:  # Aceleración saliendo de curva
            velocidad[i] = 80 + (i-65) * 4
        else:  # Última recta
            velocidad[i] = 120 + (i-75) * 2
    
    # Simular aceleraciones longitudinales (entre -0.8g y +0.6g)
    ax_g = np.zeros(n_points)
    for i in range(1, n_points):
        dv = velocidad[i] - velocidad[i-1]
        ax_g[i] = (dv / 3.6) / 0.5 / g  # Convertir a g's
    
    # Simular aceleraciones laterales (entre -1.2g y +1.2g)
    ay_g = np.zeros(n_points)
    # Curva 1 (t=20-30s)
    ay_g[40:60] = 0.8 * np.sin(np.linspace(0, np.pi, 20))
    # Curva 2 (t=50-65s) 
    ay_g[100:130] = -1.0 * np.sin(np.linspace(0, np.pi, 30))
    # Curva 3 (t=70-80s)
    ay_g[140:160] = 0.9 * np.sin(np.linspace(0, np.pi, 20))
    
    # Simular RPM (relacionado con velocidad y marchas)
    rpm = np.zeros(n_points)
    for i in range(n_points):
        if velocidad[i] < 50:
            rpm[i] = 3500 + velocidad[i] * 40
        elif velocidad[i] < 100:
            rpm[i] = 5500 + (velocidad[i] - 50) * 20
        else:
            rpm[i] = 6500 + (velocidad[i] - 100) * 10
    rpm = np.clip(rpm, 2000, 7800)
    
    # Simular marchas (basado en velocidad)
    marcha = np.zeros(n_points, dtype=int)
    for i in range(n_points):
        if velocidad[i] < 40:
            marcha[i] = 2
        elif velocidad[i] < 70:
            marcha[i] = 3
        elif velocidad[i] < 100:
            marcha[i] = 4
        else:
            marcha[i] = 5
    
    return pd.DataFrame({
        'tiempo': tiempo,
        'velocidad': np.clip(velocidad, 0, 180),
        'ax_g': np.clip(ax_g, -0.8, 0.6),
        'ay_g': np.clip(ay_g, -1.2, 1.2),
        'rpm': rpm,
        'marcha': marcha
    })

# -------------------------------
# MODELO DE POTENCIA GA16DE (NISSAN V16)
# -------------------------------
def potencia_motor_ga16de(rpm):
    """Curva de potencia realista del GA16DE para Nissan V16"""
    # Puntos característicos del GA16DE (110 hp @ 6000 rpm)
    rpm_points = [800, 1500, 2500, 3500, 4500, 5500, 6500, 7200, 7800]
    potencia_hp = [25, 45, 70, 90, 105, 110, 108, 104, 98]
    potencia_kw = [hp * 0.7457 for hp in potencia_hp]  # Convertir a kW
    
    return np.interp(rpm, rpm_points, potencia_kw)

# -------------------------------
# MODELO TÉRMICO AVANZADO
# -------------------------------
class ModeloTermicoAvanzado:
    def __init__(self):
        self.temperatura = 35.0  # Temperatura inicial (neumáticos fríos)
        self.temperaturas_history = []
        
        # Coeficientes de calentamiento para semi-slick
        self.k_calentamiento_acel = 1.2  # °C/g por segundo
        self.k_calentamiento_freno = 1.8  # °C/g por segundo  
        self.k_calentamiento_lateral = 0.6  # °C/g por segundo
        self.k_enfriamiento = 0.15  # °C por segundo
        
    def actualizar_temperatura(self, ax_g, ay_g, dt):
        """Actualiza temperatura considerando todas las aceleraciones"""
        
        # Calentamiento por aceleración longitudinal
        if ax_g > 0:  # Aceleración
            delta_T = ax_g * self.k_calentamiento_acel * dt
        elif ax_g < 0:  # Frenado
            delta_T = abs(ax_g) * self.k_calentamiento_freno * dt
        else:
            delta_T = 0
        
        # Calentamiento por aceleración lateral
        delta_T += abs(ay_g) * self.k_calentamiento_lateral * dt
        
        # Enfriamiento natural
        delta_T -= self.k_enfriamiento * dt
        
        self.temperatura += delta_T
        
        # Límites físicos realistas
        self.temperatura = max(20.0, min(115.0, self.temperatura))
        self.temperaturas_history.append(self.temperatura)
        
        return self.temperatura
    
    def mu_con_temperatura(self):
        """Modelo realista de μ vs temperatura para semi-slick"""
        T_optima = 85.0  # Temperatura óptima
        ancho_curva = 22.0  # Ancho de la curva
        
        if self.temperatura < 40:  # Zona fría - μ bajo
            return μ_max_base * 0.7 * (1 + (self.temperatura - 20) / 40)
        else:  # Curva de campana
            delta_T = self.temperatura - T_optima
            factor_temp = np.exp(-(delta_T / ancho_curva)**2)
            return μ_max_base * factor_temp

# -------------------------------
# MODELO DE ÁNGULOS PARA V16
# -------------------------------
class ModeloAngulosV16:
    def __init__(self):
        # Coeficientes optimizados para V16
        self.k_alfa_por_ay = 0.07  # rad/g - Suspensión deportiva
        self.k_gamma_por_ay = 0.7 * np.pi/180  # rad/g
        self.k_alfa_por_vel = 0.001  # Efecto de velocidad
        
    def calcular_angulos(self, ay_g, velocidad_kmh, ax_g):
        """Modelo avanzado de ángulos para V16"""
        
        # Ángulo de deriva base
        alfa_base = ay_g * self.k_alfa_por_ay
        
        # Efectos de velocidad y aceleración longitudinal
        factor_vel = 1.0 - min(0.3, velocidad_kmh * self.k_alfa_por_vel)
        factor_acel = 1.0 + abs(ax_g) * 0.3  # Más deriva bajo aceleración
        
        alfa = alfa_base * factor_vel * factor_acel
        
        # Ángulo de camber (más sensible en V16)
        gamma_base = ay_g * self.k_gamma_por_ay
        gamma = gamma_base * (1.0 + abs(ay_g) * 0.4)  # No lineal
        
        # Limitar ángulos físicamente posibles
        alfa = np.clip(alfa, -0.12, 0.12)  # ±6.9°
        gamma = np.clip(gamma, -gamma_max, gamma_max)
        
        return alfa, gamma

# -------------------------------
# FUNCIONES DE CÁLCULO
# -------------------------------
def calcular_cargas_normales(ax_g, ay_g):
    """Cálculo preciso de cargas normales para V16"""
    ax = ax_g * g
    ay = ay_g * g
    L = a + b
    
    # Cargas estáticas
    FzF0 = m * g * b / L
    FzR0 = m * g * a / L
    
    # Transferencia longitudinal
    dF_long = m * ax * h_cg / L
    FzF = max(0, FzF0 - dF_long)
    FzR = max(0, FzR0 + dF_long)
    
    # Transferencia lateral (mejorada)
    dF_lat_front = (m * ay * h_cg) * (b/L) / tf * 1.1  # Factor V16
    dF_lat_rear = (m * ay * h_cg) * (a/L) / tr * 0.9
    
    # Distribución entre ruedas
    Fz_FL = max(0, FzF/2 - dF_lat_front/2)
    Fz_FR = max(0, FzF/2 + dF_lat_front/2)
    Fz_RL = max(0, FzR/2 - dF_lat_rear/2)
    Fz_RR = max(0, FzR/2 + dF_lat_rear/2)
    
    return {
        'FL': Fz_FL, 'FR': Fz_FR, 
        'RL': Fz_RL, 'RR': Fz_RR,
        'total': FzF + FzR,
        'delantera': FzF,
        'trasera': FzR
    }

def calcular_resistencias_v16(v_kmh):
    """Resistencias específicas para V16"""
    v_ms = v_kmh / 3.6
    # Resistencia aerodinámica (Cd y A optimizados para V16)
    Cd = 0.32  # Coeficiente de arrastre bajo
    A = 1.75   # Área frontal reducida
    Fa = 0.5 * 1.225 * Cd * A * v_ms**2
    
    # Resistencia a la rodadura (neumáticos de competición)
    Crr = 0.03  # Coeficiente reducido
    Fr = Crr * m * g
    
    return Fa + Fr

# -------------------------------
# ANÁLISIS COMPLETO V16
# -------------------------------
def analisis_completo_v16():
    """Análisis completo para Nissan V16"""
    
    # Simular datos de carrera
    df_datos = simular_datos_carrera_v16()
    
    # Inicializar modelos
    modelo_termico = ModeloTermicoAvanzado()
    modelo_angulos = ModeloAngulosV16()
    
    resultados = []
    
    for i in range(len(df_datos)):
        # Datos del instante
        tiempo = df_datos['tiempo'].iloc[i]
        v_kmh = df_datos['velocidad'].iloc[i]
        v_ms = v_kmh / 3.6
        ax_g = df_datos['ax_g'].iloc[i]
        ay_g = df_datos['ay_g'].iloc[i]
        rpm = df_datos['rpm'].iloc[i]
        marcha = df_datos['marcha'].iloc[i]
        
        # 1. Actualizar modelo térmico
        dt = 0.5  # Paso de tiempo constante
        temperatura = modelo_termico.actualizar_temperatura(ax_g, ay_g, dt)
        μ_actual = modelo_termico.mu_con_temperatura()
        
        # 2. Calcular ángulos de deriva y camber
        alfa, gamma = modelo_angulos.calcular_angulos(ay_g, v_kmh, ax_g)
        
        # 3. Calcular cargas normales
        cargas = calcular_cargas_normales(ax_g, ay_g)
        N_ruedas = {'FL': cargas['FL'], 'FR': cargas['FR'], 
                   'RL': cargas['RL'], 'RR': cargas['RR']}
        
        # 4. Calcular fuerza lateral
        Y_o_total = 0
        for rueda, N in N_ruedas.items():
            if rueda in ['FL', 'RL']:
                alfa_rueda = alfa
                gamma_rueda = gamma
            else:
                alfa_rueda = -alfa
                gamma_rueda = -gamma
            
            Y_o = k_alfa * N * alfa_rueda + k_gamma * N * gamma_rueda
            Y_o_total += Y_o
        
        # 5. Calcular fuerzas longitudinales
        F_resistencia = calcular_resistencias_v16(v_kmh)
        F_aceleracion = m * ax_g * g
        F_total_required = F_resistencia + F_aceleracion
        
        # 6. Potencias
        P_required = F_total_required * v_ms / 1000
        P_disponible = potencia_motor_ga16de(rpm)
        
        # 7. Slip ratio y fuerza longitudinal
        F_trac_max = μ_actual * cargas['delantera']
        F_trac_actual = min(F_total_required, F_trac_max)
        K_actual = (F_trac_actual / cargas['delantera']) / k_K if cargas['delantera'] > 0 else 0
        X_o = k_K * cargas['delantera'] * K_actual
        
        # 8. Fuerza total y μ utilizado
        F_total_neumatico = np.sqrt(X_o**2 + Y_o_total**2)
        μ_utilizado = F_total_neumatico / cargas['total'] if cargas['total'] > 0 else 0
        
        # 9. Guardar resultados
        resultado = {
            'tiempo': tiempo,
            'velocidad_kmh': v_kmh,
            'rpm': rpm,
            'marcha': marcha,
            'ax_g': ax_g,
            'ay_g': ay_g,
            'temperatura_C': round(temperatura, 1),
            'mu_actual': round(μ_actual, 3),
            'alfa_grados': round(alfa * 180/np.pi, 2),
            'gamma_grados': round(gamma * 180/np.pi, 2),
            'N_total_N': round(cargas['total'], 1),
            'K_actual': round(K_actual, 4),
            'K_porcentaje': round(K_actual * 100, 1),
            'X_o_N': round(X_o, 1),
            'Y_o_N': round(Y_o_total, 1),
            'F_total_N': round(F_total_neumatico, 1),
            'P_req_kW': round(P_required, 1),
            'P_disp_kW': round(P_disponible, 1),
            'mu_utilizado': round(μ_utilizado, 3),
            'margen_seguridad': round((μ_actual - μ_utilizado)/μ_actual*100, 1)
        }
        
        resultados.append(resultado)
    
    return pd.DataFrame(resultados)

# -------------------------------
# VISUALIZACIÓN COMPLETA
# -------------------------------
def visualizacion_completa_v16(df):
    """Visualización completa para V16"""
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. Velocidad y RPM
    axs[0,0].plot(df['tiempo'], df['velocidad_kmh'], 'b-', label='Velocidad [km/h]')
    axs[0,0].set_ylabel('Velocidad [km/h]', color='b')
    axs[0,0].tick_params(axis='y', labelcolor='b')
    axs_twin = axs[0,0].twinx()
    axs_twin.plot(df['tiempo'], df['rpm'], 'r-', label='RPM')
    axs_twin.set_ylabel('RPM', color='r')
    axs_twin.tick_params(axis='y', labelcolor='r')
    axs[0,0].set_title('Velocidad y RPM')
    
    # 2. Temperatura y μ
    axs[0,1].plot(df['tiempo'], df['temperatura_C'], 'g-', label='Temperatura [°C]')
    axs[0,1].axhline(y=85, color='g', linestyle='--', alpha=0.7, label='Óptima (85°C)')
    axs[0,1].set_ylabel('Temperatura [°C]', color='g')
    axs[0,1].tick_params(axis='y', labelcolor='g')
    axs_twin2 = axs[0,1].twinx()
    axs_twin2.plot(df['tiempo'], df['mu_actual'], 'purple', label='μ Disponible')
    axs_twin2.set_ylabel('Coeficiente de Fricción', color='purple')
    axs_twin2.tick_params(axis='y', labelcolor='purple')
    axs[0,1].set_title('Temperatura y Adherencia')
    
    # 3. Slip Ratio
    axs[1,0].plot(df['tiempo'], df['K_porcentaje'], 'orange', label='Slip Ratio [%]')
    axs[1,0].axhline(y=10, color='green', linestyle='--', label='Óptimo (10%)')
    axs[1,0].axhline(y=20, color='red', linestyle='--', label='Límite (20%)')
    axs[1,0].set_ylabel('Slip Ratio [%]')
    axs[1,0].set_title('Slip Ratio Longitudinal')
    axs[1,0].legend()
    axs[1,0].grid(True, alpha=0.3)
    
    # 4. Ángulos de Deriva
    axs[1,1].plot(df['tiempo'], df['alfa_grados'], 'blue', label='Deriva [°]')
    axs[1,1].plot(df['tiempo'], df['gamma_grados'], 'red', label='Camber [°]')
    axs[1,1].set_ylabel('Ángulos [°]')
    axs[1,1].set_title('Ángulos de Deriva y Camber')
    axs[1,1].legend()
    axs[1,1].grid(True, alpha=0.3)
    
    # 5. Fuerzas
    axs[2,0].plot(df['tiempo'], df['X_o_N'], 'b-', label='Fuerza Longitudinal (X_o)')
    axs[2,0].plot(df['tiempo'], df['Y_o_N'], 'r-', label='Fuerza Lateral (Y_o)')
    axs[2,0].set_ylabel('Fuerza [N]')
    axs[2,0].set_title('Fuerzas Longitudinales y Laterales')
    axs[2,0].legend()
    axs[2,0].grid(True, alpha=0.3)
    
    # 6. Utilización de μ
    axs[2,1].plot(df['tiempo'], df['mu_utilizado'], 'black', label='μ Utilizado')
    axs[2,1].plot(df['tiempo'], df['mu_actual'], 'gray', linestyle='--', label='μ Disponible')
    axs[2,1].fill_between(df['tiempo'], df['mu_utilizado'], df['mu_actual'], 
                         where=df['mu_utilizado'] <= df['mu_actual'],
                         color='green', alpha=0.3, label='Margen Seguridad')
    axs[2,1].set_ylabel('Coeficiente de Fricción')
    axs[2,1].set_title('Utilización del Adherencia')
    axs[2,1].legend()
    axs[2,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analisis_v16_completo.png', dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------------
# ANÁLISIS DE PERFORMANCE
# -------------------------------
def analisis_performance_v16(df):
    """Análisis de performance para V16"""
    
    print("ANÁLISIS DE PERFORMANCE - NISSAN V16")
    print("="*50)
    
    # Estadísticas clave
    print(f"Temperatura promedio: {df['temperatura_C'].mean():.1f}°C")
    print(f"Temperatura máxima: {df['temperatura_C'].max():.1f}°C")
    print(f"μ promedio: {df['mu_actual'].mean():.3f}")
    print(f"Slip ratio promedio: {df['K_porcentaje'].mean():.1f}%")
    
    # Eficiencia
    tiempo_optimo = len(df[(df['K_porcentaje'] > 8) & (df['K_porcentaje'] < 12)]) / len(df) * 100
    print(f"Tiempo en slip ratio óptimo: {tiempo_optimo:.1f}%")
    
    # Seguridad
    margen_promedio = df['margen_seguridad'].mean()
    print(f"Margen de seguridad promedio: {margen_promedio:.1f}%")
    
    # Puntos críticos
    puntos_criticos = df[df['margen_seguridad'] < 5]
    if not puntos_criticos.empty:
        print(f"\n⚠️  PUNTOS CRÍTICOS DETECTADOS ({len(puntos_criticos)}):")
        for _, punto in puntos_criticos.head(3).iterrows():
            print(f"  t={punto['tiempo']}s: μ_utilizado={punto['mu_utilizado']:.3f}")
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    if df['temperatura_C'].mean() < 75:
        print("  - Pre-calentar más los neumáticos")
    if tiempo_optimo < 60:
        print("  - Mejorar control de aceleración para slip ratio óptimo")
    if margen_promedio > 25:
        print("  - Puedes ser más agresivo en la conducción")

# -------------------------------
# EJECUCIÓN PRINCIPAL
# -------------------------------
if __name__ == "__main__":
    print("SIMULACIÓN COMPLETA NISSAN V16")
    print("="*40)
    
    # Ejecutar análisis completo
    df_resultados = analisis_completo_v16()
    
    # Mostrar análisis
    analisis_performance_v16(df_resultados)
    
    # Visualización
    visualizacion_completa_v16(df_resultados)
    
    # Guardar resultados
    df_resultados.to_csv('simulacion_v16_completa.csv', index=False)
    print(f"\nResultados guardados en 'simulacion_v16_completa.csv'")
    print("Gráficos guardados como 'analisis_v16_completo.png'")