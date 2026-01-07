import matplotlib.pyplot as plt
import numpy as np
from reliability.Fitters import Fit_Weibull_2P
from reliability.Distributions import Weibull_Distribution
from reliability.Other_functions import make_right_censored_data
from reliability.Probability_plotting import plotting_positions

# ==============================================================================
# BLOQUE 1: CONFIGURACIÓN Y SIMULACIÓN DE DATOS
# (Este bloque lo borrarás en el futuro cuando cargues tu Excel real)
# ==============================================================================

# 1.1 Parámetros de la simulación (Modifica esto para probar escenarios)
beta_teorico = 3.0      # Forma (>1 es desgaste)
eta_teorico = 5000      # Vida característica (Scale)
total_equipos = 50      # Tamaño de la muestra
porcentaje_falla = 0.7  # 70% fallan, 30% son censurados (preventivos/activos)

print(f"--- GENERANDO SIMULACIÓN PARA {total_equipos} EQUIPOS ---")

# 1.2 Crear la distribución matemática
dist_real = Weibull_Distribution(alpha=eta_teorico, beta=beta_teorico)
datos_crudos = dist_real.random_samples(total_equipos)

# 1.3 Aplicar censura (CORREGIDO: Usamos quantile en vez de isf)
# Calculamos el tiempo donde ocurriría la falla del 70% de la población
umbral_tiempo = dist_real.quantile(porcentaje_falla) 

# Generamos los arrays de fallas y censuras
data = make_right_censored_data(datos_crudos, threshold=umbral_tiempo)
tiempos_falla = data.failures
tiempos_censurados = data.right_censored

print(f"Fallas observadas: {len(tiempos_falla)}")
print(f"Censuras (Preventivos/Operativos): {len(tiempos_censurados)}")
print("-" * 40)

# ==============================================================================
# BLOQUE 2: PROCESAMIENTO (AJUSTE DE CURVA)
# ==============================================================================

print("Ajustando modelo Weibull...")

# Ajustamos los datos simulados al modelo
# show_probability_plot=False porque haremos nuestro propio gráfico múltiple abajo
ajuste = Fit_Weibull_2P(failures=tiempos_falla, 
                        right_censored=tiempos_censurados, 
                        show_probability_plot=False, 
                        print_results=False)

# ==============================================================================
# BLOQUE 3: CÁLCULO DE KPIS (RESULTADOS)
# ==============================================================================

beta_calc = ajuste.beta
eta_calc = ajuste.alpha  # En la librería, alpha representa el parámetro de escala (Eta)

# CORREGIDO: .mean es una propiedad, no una función (sin paréntesis)
mttf_calc = ajuste.distribution.mean 

print(f"\n--- RESULTADOS DEL ANÁLISIS ---")
print(f"Beta (Forma): {beta_calc:.4f}")
print(f"Eta (Vida Característica): {eta_calc:.2f} hrs")
print(f"MTTF (Tiempo Medio para la Falla): {mttf_calc:.2f} hrs")

if beta_calc > 1:
    print(">> CONDICIÓN: Desgaste. Justifica mantenimiento preventivo.")
elif beta_calc < 1:
    print(">> CONDICIÓN: Mortalidad Infantil. Revisar montaje o calidad de repuesto.")
else:
    print(">> CONDICIÓN: Aleatorio. Solo mantenimiento correctivo/condicional.")

# ==============================================================================
# BLOQUE 4: VISUALIZACIÓN
# ==============================================================================

plt.figure(figsize=(14, 9))
plt.suptitle(f'Análisis de Confiabilidad - División Ministro Hales (Simulación)', fontsize=16)

# --- Gráfico A: Probability Plot (Linealización) ---
plt.subplot(2, 2, 1)
ajuste.distribution.CDF(label='Ajuste Weibull', color='red', linestyle='--')
# Puntos reales
x_pos, y_pos = plotting_positions(failures=tiempos_falla, right_censored=tiempos_censurados)
plt.scatter(x_pos, y_pos, c='blue', marker='o', alpha=0.6, label='Datos Reales')
plt.title('A. Linealización de Weibull')
plt.ylabel('Probabilidad Acumulada de Falla')
plt.xlabel('Tiempo (Horas)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

# --- Gráfico B: Confiabilidad R(t) ---
plt.subplot(2, 2, 2)
ajuste.distribution.SF(label='R(t) Sobrevivencia', color='green')
plt.title('B. Curva de Confiabilidad R(t)')
plt.ylabel('Probabilidad de NO fallar')
plt.xlabel('Tiempo (Horas)')
plt.grid(True, linestyle='--')
plt.legend()

# --- Gráfico C: Tasa de Falla h(t) ---
plt.subplot(2, 2, 3)
ajuste.distribution.HF(label='h(t) Hazard Rate', color='orange')
plt.title('C. Tasa de Falla (Riesgo Instantáneo)')
plt.ylabel('Fallas / Hora')
plt.xlabel('Tiempo (Horas)')
plt.grid(True, linestyle='--')
plt.legend()

# --- Gráfico D: Histograma ---
plt.subplot(2, 2, 4)
plt.hist(tiempos_falla, bins=10, color='gray', alpha=0.7, edgecolor='black', label='Fallas')
plt.axvline(mttf_calc, color='red', linestyle='dashed', linewidth=1, label=f'MTTF: {int(mttf_calc)}')
plt.title('D. Distribución de Fallas')
plt.xlabel('Tiempo (Horas)')
plt.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuste para que no choque con el título superior
plt.show()