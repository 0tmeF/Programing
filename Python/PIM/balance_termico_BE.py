import numpy as np
import matplotlib.pyplot as plt

class BalanceTermicoBancoEnsayo:
    def __init__(self):
        # Propiedades térmicas de los materiales (valores típicos)
        self.propiedades = {
            'lana_mineral': {
                'conductividad': 0.04,  # W/m·K
                'densidad': 30,         # kg/m³
                'calor_especifico': 840 # J/kg·K
            },
            'aluminio': {
                'conductividad': 237,   # W/m·K
                'densidad': 2700,       # kg/m³
                'calor_especifico': 900 # J/kg·K
            },
            'caucho': {
                'conductividad': 0.16,  # W/m·K
                'densidad': 1100,       # kg/m³
                'calor_especifico': 1670 # J/kg·K
            },
            'aire': {
                'conductividad': 0.026, # W/m·K
                'densidad': 1.2,        # kg/m³
                'calor_especifico': 1005 # J/kg·K
            }
        }
    
    def calcular_balance(self, 
                        temp_inicial=50,     # °C
                        temp_final=110,      # °C
                        temp_ambiente=25,    # °C
                        espesor_aislante=0.18, # m (18 cm)
                        tiempo_calentamiento=3600, # s (1 hora)
                        area_apertura=0.06,  # m² (20x30 cm)
                        velocidad_viento=1.0): # m/s
        
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
        
        # 1. CALOR PARA CALENTAR LA PROBETA
        masa_probeta = volumen_probeta * self.propiedades['caucho']['densidad']
        calor_probeta = masa_probeta * self.propiedades['caucho']['calor_especifico'] * (temp_final - temp_inicial)
        
        # 2. CALOR PARA CALENTAR EL AIRE INTERNO
        masa_aire = volumen_aire * self.propiedades['aire']['densidad']
        calor_aire = masa_aire * self.propiedades['aire']['calor_especifico'] * (temp_final - temp_inicial)
        
        # 3. PÉRDIDAS POR CONDUCCIÓN A TRAVÉS DE LAS PAREDES
        # Resistencia térmica de la pared (4 capas de aluminio + lana mineral)
        # Espesor de cada capa de aluminio (asumido)
        espesor_aluminio = 0.0001  # 0.1 mm
        
        R_aluminio = espesor_aluminio / self.propiedades['aluminio']['conductividad']
        R_aislante = espesor_aislante / self.propiedades['lana_mineral']['conductividad']
        
        R_total_pared = 4 * R_aluminio + R_aislante
        U_pared = 1 / R_total_pared  # Coeficiente global de transferencia
        
        # Diferencia de temperatura promedio durante el calentamiento
        delta_T_promedio = ((temp_final + temp_inicial) / 2) - temp_ambiente
        
        # Pérdidas por conducción
        perdidas_conduccion = U_pared * area_paredes * delta_T_promedio * tiempo_calentamiento
        
        # 4. PÉRDIDAS POR LA APERTURA (convección y radiación)
        # Coeficiente de convección natural (simplificado)
        h_conveccion = 5.0  # W/m²·K para convección natural moderada
        
        # Pérdidas por convección a través de la apertura
        perdidas_conveccion = h_conveccion * area_apertura * delta_T_promedio * tiempo_calentamiento
        
        # Pérdidas por radiación (ley de Stefan-Boltzmann)
        sigma = 5.67e-8  # W/m²·K⁴
        emisividad = 0.9  # Para superficie no pulida
        
        T_int_prom = (temp_final + temp_inicial) / 2 + 273.15  # K
        T_ext = temp_ambiente + 273.15  # K
        
        perdidas_radiacion = (emisividad * sigma * area_apertura * 
                             (T_int_prom**4 - T_ext**4) * tiempo_calentamiento)
        
        # 5. CALOR TOTAL REQUERIDO
        calor_total = (calor_probeta + calor_aire + 
                      perdidas_conduccion + perdidas_conveccion + perdidas_radiacion)
        
        # Potencia promedio requerida
        potencia_promedio = calor_total / tiempo_calentamiento
        
        # Resultados
        resultados = {
            'calor_total': calor_total,
            'potencia_promedio': potencia_promedio,
            'calor_probeta': calor_probeta,
            'calor_aire': calor_aire,
            'perdidas_conduccion': perdidas_conduccion,
            'perdidas_conveccion': perdidas_conveccion,
            'perdidas_radiacion': perdidas_radiacion,
            'masa_probeta': masa_probeta,
            'masa_aire': masa_aire,
            'U_pared': U_pared
        }
        
        return resultados
    
    def simular_variaciones(self, parametro, valores, temp_final=110):
        """
        Simula variaciones de un parámetro específico
        """
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
            else:
                continue
            
            resultados_variacion.append(resultados)
        
        return resultados_variacion
    
    def graficar_resultados(self, resultados):
        """
        Grafica los resultados del balance térmico
        """
        labels = ['Calor Probeta', 'Calor Aire', 'Pérdidas Conducción', 
                 'Pérdidas Convección', 'Pérdidas Radiación']
        valores = [
            resultados['calor_probeta'] / 1000,
            resultados['calor_aire'] / 1000,
            resultados['perdidas_conduccion'] / 1000,
            resultados['perdidas_conveccion'] / 1000,
            resultados['perdidas_radiacion'] / 1000
        ]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico de torta
        ax1.pie(valores, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Distribución del Calor Requerido')
        
        # Gráfico de barras
        bars = ax2.bar(labels, valores)
        ax2.set_title('Contribuciones al Balance Térmico')
        ax2.set_ylabel('Energía (kJ)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Añadir valores en las barras
        for bar, valor in zip(bars, valores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{valor:.1f} kJ', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def imprimir_resultados(self, resultados):
        """
        Imprime los resultados de forma legible
        """
        print("=" * 60)
        print("BALANCE TÉRMICO - BANCO DE ENSAYOS")
        print("=" * 60)
        print(f"Calor total requerido: {resultados['calor_total']/1000:.2f} kJ")
        print(f"Potencia promedio: {resultados['potencia_promedio']:.2f} W")
        print(f"Tiempo de calentamiento: {3600/3600:.1f} h")
        print("\nDETALLE DE CONTRIBUCIONES:")
        print(f"- Calor para probeta: {resultados['calor_probeta']/1000:.2f} kJ")
        print(f"- Calor para aire interno: {resultados['calor_aire']/1000:.2f} kJ")
        print(f"- Pérdidas por conducción: {resultados['perdidas_conduccion']/1000:.2f} kJ")
        print(f"- Pérdidas por convección: {resultados['perdidas_conveccion']/1000:.2f} kJ")
        print(f"- Pérdidas por radiación: {resultados['perdidas_radiacion']/1000:.2f} kJ")
        print(f"\nMasa de la probeta: {resultados['masa_probeta']*1000:.2f} g")
        print(f"Coeficiente U de paredes: {resultados['U_pared']:.4f} W/m²·K")

# EJEMPLO DE USO
if __name__ == "__main__":
    # Crear instancia del balance térmico
    balance = BalanceTermicoBancoEnsayo()
    
    # Calcular balance base
    print("CÁLCULO BASE:")
    resultados_base = balance.calcular_balance()
    balance.imprimir_resultados(resultados_base)
    balance.graficar_resultados(resultados_base)
    
    # Simular variaciones de espesor de aislante
    print("\n" + "="*60)
    print("ANÁLISIS DE SENSIBILIDAD - ESPESOR DEL AISLANTE")
    print("="*60)
    
    espesores = [0.10, 0.15, 0.20, 0.25]  # m
    resultados_espesores = balance.simular_variaciones('espesor_aislante', espesores)
    
    for i, (espesor, resultado) in enumerate(zip(espesores, resultados_espesores)):
        print(f"Espesor {espesor*100:.0f} cm -> Potencia: {resultado['potencia_promedio']:.1f} W")
    
    # Simular variaciones de área de apertura
    print("\n" + "="*60)
    print("ANÁLISIS DE SENSIBILIDAD - ÁREA DE APERTURA")
    print("="*60)
    
    areas = [0.02, 0.04, 0.06, 0.08]  # m²
    resultados_areas = balance.simular_variaciones('area_apertura', areas)
    
    for i, (area, resultado) in enumerate(zip(areas, resultados_areas)):
        print(f"Área {area*10000:.0f} cm² -> Potencia: {resultado['potencia_promedio']:.1f} W")