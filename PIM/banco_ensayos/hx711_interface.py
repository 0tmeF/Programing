# -*- coding: utf-8 -*-
"""
Interface HX711 para célula de carga - Banco de Ensayos
"""
import time
import json
import numpy as np
from datetime import datetime

class HX711Interface:
    """
    Interface para célula de carga HX711
    NOTA: Esta es una implementación de simulación
    Para uso real, instalar biblioteca apropiada e implementar GPIO
    """
    
    def __init__(self, dout_pin=5, sck_pin=6, gain=128):
        self.dout_pin = dout_pin
        self.sck_pin = sck_pin
        self.gain = gain
        
        # Parámetros de calibración
        self.offset = 0
        self.scale_factor = 1.0
        self.is_calibrated = False
        
        # Estado del sensor
        self.last_reading = 0
        self.reading_count = 0
        
        print(f"🔧 HX711 Interface inicializada (Pins: DOUT={dout_pin}, SCK={sck_pin})")
    
    def initialize_sensor(self):
        """Inicializa el sensor HX711"""
        try:
            # Simular inicialización (en hardware real: configurar GPIO)
            print("✅ Sensor HX711 inicializado (simulación)")
            return True
        except Exception as e:
            print(f"❌ Error inicializando sensor: {e}")
            return False
    
    def read_raw(self):
        """Lee valor raw del sensor HX711"""
        # SIMULACIÓN - Reemplazar con implementación real usando RPi.GPIO
        try:
            # Simular lectura con ruido y deriva
            base_value = 850000 + np.random.randint(-1000, 1000)
            # Simular deriva térmica leve
            thermal_drift = np.sin(time.time() / 100) * 500
            raw_value = base_value + thermal_drift
            
            self.last_reading = raw_value
            self.reading_count += 1
            
            return raw_value
            
        except Exception as e:
            print(f"❌ Error leyendo sensor: {e}")
            return None
    
    def calibrate_tare(self, samples=10):
        """Calibración de tarado (cero)"""
        print("🔧 Calibrando tarado...")
        readings = []
        
        for i in range(samples):
            raw = self.read_raw()
            if raw is not None:
                readings.append(raw)
            time.sleep(0.1)
        
        if readings:
            self.offset = np.mean(readings)
            print(f"✅ Tarado calibrado: offset = {self.offset:.0f}")
            return True
        else:
            print("❌ Error en calibración de tarado")
            return False
    
    def calibrate_scale(self, known_weight_kg):
        """Calibración de escala con peso conocido"""
        known_weight_newtons = known_weight_kg * 9.81
        print(f"🔧 Calibrando escala con {known_weight_kg} kg ({known_weight_kg * 9.81:.1f} N)")
        
        # Leer con peso aplicado
        raw_with_weight = self.read_raw()
        if raw_with_weight is None:
            print("❌ Error leyendo sensor con peso")
            return False
        
        # Calcular factor de escala
        raw_difference = raw_with_weight - self.offset
        self.scale_factor = raw_difference / known_weight_newtons
        
        self.is_calibrated = True
        print(f"✅ Escala calibrada: factor = {self.scale_factor:.3f}")
        return True
    
    def get_weight(self, samples=5):
        """Obtiene peso en Newtons promediando múltiples lecturas"""
        if not self.is_calibrated:
            print("⚠️  Sensor no calibrado. Usando calibración por defecto.")
        
        readings = []
        for _ in range(samples):
            raw = self.read_raw()
            if raw is not None:
                weight = (raw - self.offset) / self.scale_factor
                readings.append(weight)
            time.sleep(0.05)
        
        if readings:
            avg_weight = np.mean(readings)
            std_dev = np.std(readings)
            return avg_weight, std_dev
        else:
            return None, None
    
    def continuous_reading(self, duration_sec=30, sample_rate=2):
        """Lectura continua por tiempo determinado"""
        print(f"📊 Iniciando lectura continua por {duration_sec} segundos...")
        
        start_time = time.time()
        readings = []
        
        try:
            while time.time() - start_time < duration_sec:
                weight, std = self.get_weight(samples=3)
                if weight is not None:
                    readings.append({
                        'timestamp': time.time(),
                        'weight_n': weight,
                        'std_dev': std
                    })
                    print(f"  Fuerza: {weight:7.2f} N ± {std:.2f}")
                
                time.sleep(1.0 / sample_rate)
                
        except KeyboardInterrupt:
            print("\n🛑 Lectura interrumpida por usuario")
        
        return readings

# Ejemplo de uso y pruebas
if __name__ == "__main__":
    # Crear instancia del sensor
    hx711 = HX711Interface(dout_pin=5, sck_pin=6)
    
    # Inicializar sensor
    if hx711.initialize_sensor():
        # Calibrar tarado (sin peso)
        print("\n1. CALIBRACIÓN DE TARADO")
        input("   Retira todo peso y presiona Enter...")
        hx711.calibrate_tare()
        
        # Calibrar escala (con peso conocido)
        print("\n2. CALIBRACIÓN DE ESCALA")
        peso_kg = float(input("   Ingresa peso conocido en kg: "))
        input("   Coloca el peso y presiona Enter...")
        hx711.calibrate_scale(peso_kg)
        
        # Lectura continua de prueba
        print("\n3. LECTURA CONTINUA DE PRUEBA (10 segundos)")
        lecturas = hx711.continuous_reading(duration_sec=10)
        
        # Estadísticas
        if lecturas:
            fuerzas = [l['weight_n'] for l in lecturas]
            print(f"\n📈 Estadísticas:")
            print(f"   Mínimo: {min(fuerzas):.2f} N")
            print(f"   Máximo: {max(fuerzas):.2f} N") 
            print(f"   Promedio: {np.mean(fuerzas):.2f} N")
            print(f"   Desviación: {np.std(fuerzas):.2f} N")
        
        print("\n✅ Prueba completada")