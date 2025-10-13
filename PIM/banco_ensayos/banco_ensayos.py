# -*- coding: utf-8 -*-
"""
BANCO DE ENSAYOS - Caracterización de coeficiente de fricción
Sistema para Nissan Sentra B16
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

class BancoEnsayos:
    def __init__(self):
        self.datos_calibracion = []
        self.resultados = []
        
    def calibrar_sensor_fuerza(self, peso_conocido_kg):
        """
        Calibra el sensor de fuerza con peso conocido
        """
        print(f"🔧 Calibrando con peso conocido: {peso_conocido_kg} kg")
        fuerza_newtons = peso_conocido_kg * 9.81
        print(f"   Fuerza equivalente: {fuerza_newtons:.2f} N")
        
        # Simular lectura del sensor (reemplazar con HX711 real)
        lectura_sensor = self.simular_lectura_sensor(fuerza_newtons)
        factor_calibracion = lectura_sensor / fuerza_newtons
        
        calibracion = {
            'fecha': datetime.now().isoformat(),
            'peso_kg': peso_conocido_kg,
            'fuerza_n': fuerza_newtons,
            'lectura_sensor': lectura_sensor,
            'factor_calibracion': factor_calibracion
        }
        
        self.datos_calibracion.append(calibracion)
        return factor_calibracion
    
    def simular_lectura_sensor(self, fuerza_real):
        """
        Simula la lectura del sensor HX711
        En producción, reemplazar con lectura real del hardware
        """
        # Simular ruido de medición
        ruido = np.random.normal(0, fuerza_real * 0.02)  # 2% de ruido
        return fuerza_real + ruido
    
    def medir_coeficiente_friccion(self, temperatura_c, fuerza_normal_n, num_mediciones=5):
        """
        Mide el coeficiente de fricción para una temperatura dada
        """
        print(f"🌡️  Midendo μ a {temperatura_c}°C con Fn={fuerza_normal_n}N")
        
        mediciones_friccion = []
        
        for i in range(num_mediciones):
            # Simular medición de fuerza de fricción
            fuerza_friccion = self.simular_fuerza_friccion(fuerza_normal_n, temperatura_c)
            coef_friccion = fuerza_friccion / fuerza_normal_n
            
            medicion = {
                'medicion': i + 1,
                'fuerza_friccion_n': fuerza_friccion,
                'coef_friccion': coef_friccion
            }
            mediciones_friccion.append(medicion)
            
            print(f"   Medicion {i+1}: μ = {coef_friccion:.3f}")
        
        # Calcular promedio
        coef_promedio = np.mean([m['coef_friccion'] for m in mediciones_friccion])
        desviacion = np.std([m['coef_friccion'] for m in mediciones_friccion])
        
        resultado = {
            'temperatura_c': temperatura_c,
            'fuerza_normal_n': fuerza_normal_n,
            'coef_friccion_promedio': coef_promedio,
            'desviacion_estandar': desviacion,
            'mediciones': mediciones_friccion,
            'timestamp': datetime.now().isoformat()
        }
        
        self.resultados.append(resultado)
        return resultado
    
    def simular_fuerza_friccion(self, fuerza_normal, temperatura):
        """
        Simula la fuerza de fricción basada en modelo térmico de neumáticos
        """
        # Modelo simplificado: μ máximo alrededor de 70-80°C para semi-slick
        temp_optima = 75
        mu_maximo = 1.2
        mu_minimo = 0.8
        
        # Curva gaussiana centrada en temperatura óptima
        delta_temp = temperatura - temp_optima
        mu = mu_maximo * np.exp(-0.5 * (delta_temp / 25) ** 2)
        
        # Asegurar valor mínimo
        mu = max(mu, mu_minimo)
        
        # Añadir ruido de medición
        ruido = np.random.normal(0, 0.05)
        mu += ruido
        
        return mu * fuerza_normal
    
    def generar_curva_friccion(self, temperaturas, fuerza_normal=200):
        """
        Genera curva completa de coeficiente de fricción vs temperatura
        """
        print("📈 GENERANDO CURVA μ vs TEMPERATURA")
        print("=" * 50)
        
        for temp in temperaturas:
            self.medir_coeficiente_friccion(temp, fuerza_normal)
            
        return self.resultados
    
    def visualizar_curva(self):
        """Genera gráfico de la curva de fricción"""
        if not self.resultados:
            print("No hay datos para visualizar")
            return
            
        temps = [r['temperatura_c'] for r in self.resultados]
        mus = [r['coef_friccion_promedio'] for r in self.resultados]
        errores = [r['desviacion_estandar'] for r in self.resultados]
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(temps, mus, yerr=errores, fmt='o-', capsize=5, linewidth=2)
        plt.xlabel('Temperatura (°C)')
        plt.ylabel('Coeficiente de Fricción (μ)')
        plt.title('Curva de Fricción vs Temperatura - Neumáticos Semi-slick')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def guardar_resultados(self, nombre_archivo=None):
        """Guarda resultados en archivo JSON"""
        if not nombre_archivo:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo = f"resultados_banco_ensayos_{timestamp}.json"
        
        datos_exportar = {
            'metadata': {
                'vehiculo': 'Nissan Sentra B16',
                'tipo_neumatico': 'Semi-slick',
                'fecha_generacion': datetime.now().isoformat()
            },
            'calibraciones': self.datos_calibracion,
            'resultados': self.resultados
        }
        
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            json.dump(datos_exportar, f, indent=2, ensure_ascii=False)
            
        print(f"💾 Resultados guardados en: {nombre_archivo}")

# Ejemplo de uso
if __name__ == "__main__":
    banco = BancoEnsayos()
    
    # Calibrar sensor
    banco.calibrar_sensor_fuerza(10.0)  # 10 kg
    
    # Generar curva para diferentes temperaturas
    temperaturas = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    resultados = banco.generar_curva_friccion(temperaturas)
    
    # Visualizar resultados
    banco.visualizar_curva()
    banco.guardar_resultados()
    
    print("\n🎯 ANÁLISIS COMPLETADO")
    for resultado in resultados:
        print(f"  {resultado['temperatura_c']:3}°C → μ = {resultado['coef_friccion_promedio']:.3f} ± {resultado['desviacion_estandar']:.3f}")