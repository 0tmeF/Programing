import numpy as np
import matplotlib.pyplot as plt

class BalanceTermicoBancoEnsayo:
    def __init__(self):
        # Propiedades térmicas de los materiales (valores actualizados)
        self.propiedades = {
            'lana_mineral': {
                'conductividad': 0.04,    # W/m·K
                'densidad': 30,           # kg/m³
            },
            'aluminio': {
                'conductividad': 237,     # W/m·K
                'emisividad': 0.05,       # Superficie reflectiva
            },
            'caucho': {
                'conductividad': 0.16,    # W/m·K
                'densidad': 1100,         # kg/m³
                'calor_especifico': 1670, # J/kg·K
                'emisividad': 0.85,       # Caucho típico
            },
            'aire': {
                'densidad': 1.2,          # kg/m³
                'calor_especifico': 1005, # J/kg·K
            }
        }
        
        # Constante de Stefan-Boltzmann
        self.sigma = 5.67e-8  # W/m²·K⁴
    
    def calcular_balance(self, 
                        temp_inicial=50,      # °C
                        temp_final=110,       # °C
                        temp_ambiente=25,     # °C
                        espesor_aislante=0.18, # m (18 cm)
                        tiempo_calentamiento=3600, # s (1 hora)
                        area_apertura=0.06,   # m² (20x30 cm)
                        emisividad_apertura=0.9): # Superficie no reflectiva
        
        # Dimensiones de la caja (convertir a metros)
        largo = 0.40   # m
        alto = 0.35    # m
        ancho = 0.30   # m
        
        # Dimensiones de la probeta
        area_probeta = 0.25 * 0.25  # m²
        espesor_probeta = 0.01      # m
        
        # Cálculo de áreas y volúmenes
        area_paredes = 2 * (largo * alto + largo * ancho + alto * ancho)
        volumen_caja = largo * alto * ancho
        volumen_probeta = area_probeta * espesor_probeta
        volumen_aire = volumen_caja - volumen_probeta
        
        # Temperaturas en Kelvin para cálculos de radiación
        T_int_prom = ((temp_final + temp_inicial) / 2) + 273.15  # K
        T_ext = temp_ambiente + 273.15  # K
        delta_T_promedio = ((temp_final + temp_inicial) / 2) - temp_ambiente
        
        # 1. CALOR PARA CALENTAR LA PROBETA
        masa_probeta = volumen_probeta * self.propiedades['caucho']['densidad']
        calor_probeta = masa_probeta * self.propiedades['caucho']['calor_especifico'] * (temp_final - temp_inicial)
        
        # 2. CALOR PARA CALENTAR EL AIRE INTERNO
        masa_aire = volumen_aire * self.propiedades['aire']['densidad']
        calor_aire = masa_aire * self.propiedades['aire']['calor_especifico'] * (temp_final - temp_inicial)
        
        # 3. PÉRDIDAS POR CONDUCCIÓN A TRAVÉS DE LAS PAREDES
        espesor_aluminio = 0.0001  # 0.1 mm por capa
        
        R_aluminio = espesor_aluminio / self.propiedades['aluminio']['conductividad']
        R_aislante = espesor_aislante / self.propiedades['lana_mineral']['conductividad']
        
        R_total_pared = 4 * R_aluminio + R_aislante
        U_pared = 1 / R_total_pared
        
        perdidas_conduccion = U_pared * area_paredes * delta_T_promedio * tiempo_calentamiento
        
        # 4. PÉRDIDAS POR RADIACIÓN A TRAVÉS DE PAREDES (CON ALUMINIO REFLECTIVO)
        emisividad_paredes = self.propiedades['aluminio']['emisividad']
        perdidas_radiacion_paredes = (emisividad_paredes * self.sigma * area_paredes * 
                                    (T_int_prom**4 - T_ext**4) * tiempo_calentamiento)
        
        # 5. PÉRDIDAS POR LA APERTURA (convección + radiación)
        # 5.1 Convección a través de la apertura
        h_conveccion = 5.0  # W/m²·K para convección natural
        perdidas_conveccion = h_conveccion * area_apertura * delta_T_promedio * tiempo_calentamiento
        
        # 5.2 Radiación a través de la apertura
        perdidas_radiacion_apertura = (emisividad_apertura * self.sigma * area_apertura * 
                                     (T_int_prom**4 - T_ext**4) * tiempo_calentamiento)
        
        # 6. CALOR TOTAL REQUERIDO (ACTUALIZADO)
        calor_total = (calor_probeta + calor_aire + 
                      perdidas_conduccion + 
                      perdidas_radiacion_paredes +
                      perdidas_conveccion + 
                      perdidas_radiacion_apertura)
        
        # Potencia promedio requerida
        potencia_promedio = calor_total / tiempo_calentamiento
        
        # Resultados completos
        resultados = {
            'calor_total': calor_total,
            'potencia_promedio': potencia_promedio,
            'calor_probeta': calor_probeta,
            'calor_aire': calor_aire,
            'perdidas_conduccion': perdidas_conduccion,
            'perdidas_radiacion_paredes': perdidas_radiacion_paredes,
            'perdidas_conveccion': perdidas_conveccion,
            'perdidas_radiacion_apertura': perdidas_radiacion_apertura,
            'masa_probeta': masa_probeta,
            'masa_aire': masa_aire,
            'U_pared': U_pared,
            'area_paredes': area_paredes,
            'area_apertura': area_apertura
        }
        
        return resultados
    
    def comparar_con_sin_aluminio(self, temp_final=110):
        """Compara el desempeño con y sin aluminio reflectivo"""
        
        # Con aluminio (configuración actual)
        resultados_con_al = self.calcular_balance(temp_final=temp_final)
        
        # Sin aluminio (emisividad alta en paredes)
        resultados_sin_al = self.calcular_balance(
            temp_final=temp_final,
            emisividad_apertura=0.9  # Misma emisividad para paredes y apertura
        )
        
        print("=" * 70)
        print("COMPARACIÓN: CON vs SIN ALUMINIO REFLECTIVO")
        print("=" * 70)
        print(f"CON aluminio:")
        print(f"  - Potencia requerida: {resultados_con_al['potencia_promedio']:.1f} W")
        print(f"  - Pérdidas por radiación paredes: {resultados_con_al['perdidas_radiacion_paredes']/1000:.1f} kJ")
        
        print(f"\nSIN aluminio (supuesto):")
        print(f"  - Potencia requerida: {resultados_sin_al['potencia_promedio']:.1f} W") 
        print(f"  - Pérdidas por radiación paredes: {resultados_sin_al['perdidas_radiacion_paredes']/1000:.1f} kJ")
        
        ahorro = ((resultados_sin_al['potencia_promedio'] - resultados_con_al['potencia_promedio']) / 
                 resultados_sin_al['potencia_promedio'] * 100)
        print(f"\nAHORRO con aluminio: {ahorro:.1f}%")
        
        return resultados_con_al, resultados_sin_al
    
    def simular_variaciones(self, parametro, valores, temp_final=110):
        """Simula variaciones de un parámetro específico"""
        resultados_variacion = []
        
        for valor in valores:
            if parametro == 'espesor_aislante':
                resultados = self.calcular_balance(espesor_aislante=valor, temp_final=temp_final)
            elif parametro == 'area_apertura':
                resultados = self.calcular_balance(area_apertura=valor, temp_final=temp_final)
            elif parametro == 'tiempo_calentamiento':
                resultados = self.calcular_balance(tiempo_calentamiento=valor, temp_final=temp_final)
            elif parametro == 'temp_final':
                resultados = self.calcular_balance(temp_final=valor)
            elif parametro == 'emisividad_apertura':
                resultados = self.calcular_balance(emisividad_apertura=valor, temp_final=temp_final)
            else:
                continue
            
            resultados_variacion.append(resultados)
        
        return resultados_variacion
    
    def graficar_resultados(self, resultados):
        """Grafica los resultados del balance térmico"""
        labels = ['Calor Probeta', 'Calor Aire', 'Pérdidas Conducción', 
                 'Pérdidas Radiación Paredes', 'Pérdidas Convección', 'Pérdidas Radiación Apertura']
        valores = [
            resultados['calor_probeta'] / 1000,
            resultados['calor_aire'] / 1000,
            resultados['perdidas_conduccion'] / 1000,
            resultados['perdidas_radiacion_paredes'] / 1000,
            resultados['perdidas_conveccion'] / 1000,
            resultados['perdidas_radiacion_apertura'] / 1000
        ]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gráfico de torta
        ax1.pie(valores, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Distribución del Calor Requerido (CON aluminio reflectivo)')
        
        # Gráfico de barras
        bars = ax2.bar(labels, valores)
        ax2.set_title('Contribuciones al Balance Térmico')
        ax2.set_ylabel('Energía (kJ)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Añadir valores en las barras
        for bar, valor in zip(bars, valores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{valor:.1f} kJ', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def imprimir_resultados(self, resultados):
        """Imprime los resultados de forma legible"""
        print("=" * 70)
        print("BALANCE TÉRMICO - BANCO DE ENSAYOS (CON ALUMINIO REFLECTIVO)")
        print("=" * 70)
        print(f"Calor total requerido: {resultados['calor_total']/1000:.2f} kJ")
        print(f"Potencia promedio: {resultados['potencia_promedio']:.2f} W")
        print(f"Tiempo de calentamiento: {3600/3600:.1f} h")
        print("\nDETALLE DE CONTRIBUCIONES:")
        print(f"- Calor para probeta: {resultados['calor_probeta']/1000:.2f} kJ")
        print(f"- Calor para aire interno: {resultados['calor_aire']/1000:.2f} kJ")
        print(f"- Pérdidas por conducción: {resultados['perdidas_conduccion']/1000:.2f} kJ")
        print(f"- Pérdidas por radiación (paredes): {resultados['perdidas_radiacion_paredes']/1000:.2f} kJ")
        print(f"- Pérdidas por convección (apertura): {resultados['perdidas_conveccion']/1000:.2f} kJ")
        print(f"- Pérdidas por radiación (apertura): {resultados['perdidas_radiacion_apertura']/1000:.2f} kJ")
        print(f"\nMasa de la probeta: {resultados['masa_probeta']*1000:.2f} g")
        print(f"Coeficiente U de paredes: {resultados['U_pared']:.4f} W/m²·K")
        print(f"Área total paredes: {resultados['area_paredes']:.3f} m²")
        print(f"Área apertura: {resultados['area_apertura']:.3f} m²")

# EJEMPLO DE USO
if __name__ == "__main__":
    # Crear instancia del balance térmico
    balance = BalanceTermicoBancoEnsayo()
    
    # 1. Cálculo base CON aluminio reflectivo
    print("CÁLCULO BASE CON ALUMINIO REFLECTIVO:")
    resultados_base = balance.calcular_balance()
    balance.imprimir_resultados(resultados_base)
    balance.graficar_resultados(resultados_base)
    
    # 2. Comparación CON vs SIN aluminio
    resultados_con, resultados_sin = balance.comparar_con_sin_aluminio()
    
    # 3. Análisis de sensibilidad - Espesor del aislante
    print("\n" + "="*70)
    print("ANÁLISIS DE SENSIBILIDAD - ESPESOR DEL AISLANTE")
    print("="*70)
    
    espesores = [0.10, 0.15, 0.18, 0.20, 0.25]  # m
    resultados_espesores = balance.simular_variaciones('espesor_aislante', espesores)
    
    for i, (espesor, resultado) in enumerate(zip(espesores, resultados_espesores)):
        print(f"Espesor {espesor*100:.0f} cm -> Potencia: {resultado['potencia_promedio']:.1f} W")
    
    # 4. Análisis de sensibilidad - Área de apertura
    print("\n" + "="*70)
    print("ANÁLISIS DE SENSIBILIDAD - ÁREA DE APERTURA")
    print("="*70)
    
    areas = [0.02, 0.04, 0.06, 0.08, 0.10]  # m²
    resultados_areas = balance.simular_variaciones('area_apertura', areas)
    
    for i, (area, resultado) in enumerate(zip(areas, resultados_areas)):
        print(f"Área {area*10000:.0f} cm² -> Potencia: {resultado['potencia_promedio']:.1f} W")