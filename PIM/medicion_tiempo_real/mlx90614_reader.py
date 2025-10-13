# -*- coding: utf-8 -*-
"""
Lector de sensores MLX90614 para medición de temperatura en neumáticos
Configuración específica para Nissan Sentra B16
"""
import time
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import smbus2
    I2C_AVAILABLE = True
except ImportError:
    I2C_AVAILABLE = False
    logger.warning("smbus2 no disponible - ejecutando en modo simulación")

class MLX90614:
    """
    Clase para manejar sensores MLX90614 de temperatura infrarroja
    """
    
    # Registros MLX90614
    REG_TA = 0x06  # Temperatura ambiente
    REG_TOBJ1 = 0x07  # Temperatura objeto 1
    
    def __init__(self, bus=1, address=0x5A, sensor_name="Unknown"):
        self.bus_num = bus
        self.address = address
        self.sensor_name = sensor_name
        self.is_connected = False
        
        if I2C_AVAILABLE:
            try:
                self.bus = smbus2.SMBus(bus)
                self.is_connected = True
                logger.info(f"✅ Sensor {sensor_name} conectado (0x{address:02X})")
            except Exception as e:
                logger.error(f"❌ Error conectando sensor {sensor_name}: {e}")
                self.is_connected = False
        else:
            logger.warning(f"⚠️  Sensor {sensor_name} en modo simulación")
            self.is_connected = False
    
    def read_temperature(self, register=REG_TOBJ1):
        """
        Lee temperatura del sensor en grados Celsius
        """
        if not self.is_connected:
            # Modo simulación para desarrollo
            return self._simulate_temperature()
        
        try:
            # Leer 3 bytes del registro
            data = self.bus.read_i2c_block_data(self.address, register, 3)
            
            # Convertir a temperatura (datasheet MLX90614)
            temp_raw = (data[1] << 8) | data[0]
            temperature = (temp_raw * 0.02) - 273.15
            
            return round(temperature, 2)
            
        except Exception as e:
            logger.error(f"Error leyendo sensor {self.sensor_name}: {e}")
            return None
    
    def _simulate_temperature(self):
        """
        Simula lecturas de temperatura para desarrollo sin hardware
        """
        # Simular temperatura base según posición del neumático
        base_temps = {
            "Delantero Izquierdo": 65,
            "Delantero Derecho": 72,  # Más carga en curvas
            "Trasero Derecho": 58
        }
        
        base_temp = base_temps.get(self.sensor_name, 60)
        
        # Simular variaciones realistas
        import random
        variation = random.gauss(0, 2)  # Variación normal ±2°C
        simulated_temp = base_temp + variation
        
        return round(max(20, simulated_temp), 1)  # Mínimo 20°C
    
    def read_ambient_temperature(self):
        """Lee temperatura ambiente"""
        return self.read_temperature(self.REG_TA)
    
    def check_connection(self):
        """Verifica si el sensor está conectado y respondiendo"""
        if not I2C_AVAILABLE:
            return True  # En simulación siempre "conectado"
        
        try:
            temp = self.read_temperature()
            return temp is not None
        except:
            return False

def setup_tire_sensors():
    """
    Configura los 3 sensores MLX90614 para los neumáticos del Nissan Sentra B16
    """
    sensors_config = [
        {"name": "Delantero Izquierdo", "address": 0x5A},
        {"name": "Delantero Derecho", "address": 0x5B},
        {"name": "Trasero Derecho", "address": 0x5C}
    ]
    
    sensors = {}
    
    print("🔍 INICIALIZANDO SENSORES MLX90614")
    print("=" * 40)
    
    for config in sensors_config:
        sensor = MLX90614(
            bus=1,
            address=config["address"],
            sensor_name=config["name"]
        )
        
        sensors[config["name"]] = sensor
        
        # Verificar conexión
        if sensor.check_connection():
            print(f"  ✅ {config['name']}: Conectado (0x{config['address']:02X})")
        else:
            print(f"  ❌ {config['name']}: No responde (0x{config['address']:02X})")
    
    return sensors

def continuous_monitoring(sensors, duration_min=5, sample_interval=2):
    """
    Monitoreo continuo de temperaturas
    """
    print(f"\n📊 INICIANDO MONITOREO ({duration_min} minutos)")
    print("=" * 50)
    
    start_time = time.time()
    end_time = start_time + (duration_min * 60)
    
    try:
        while time.time() < end_time:
            timestamp = datetime.now().strftime("%H:%M:%S")
            readings = {}
            
            # Leer todos los sensores
            for name, sensor in sensors.items():
                temp = sensor.read_temperature()
                if temp is not None:
                    readings[name] = temp
            
            # Mostrar resultados
            print(f"🕒 {timestamp} ", end="")
            for name, temp in readings.items():
                # Abreviar nombres para display
                short_name = name.replace("Delantero", "Del").replace("Izquierdo", "Izq").replace("Derecho", "Der")
                print(f"| {short_name}: {temp:5.1f}°C ", end="")
            print()
            
            time.sleep(sample_interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoreo interrumpido por usuario")
    
    print("✅ Monitoreo finalizado")

# Función principal para pruebas
if __name__ == "__main__":
    print("🚗 SISTEMA DE TEMPERATURA - NISSAN SENTRA B16")
    print("Sensores MLX90614 para neumáticos")
    print()
    
    # Configurar sensores
    sensors = setup_tire_sensors()
    
    # Verificar que al menos un sensor esté funcionando
    working_sensors = [name for name, sensor in sensors.items() if sensor.check_connection()]
    
    if not working_sensors:
        print("❌ No se detectaron sensores funcionando. Verificar conexiones I2C.")
        exit(1)
    
    print(f"\n🎯 Sensores operativos: {len(working_sensors)}/3")
    
    # Iniciar monitoreo
    try:
        continuous_monitoring(sensors, duration_min=10, sample_interval=2)
    except Exception as e:
        logger.error(f"Error en monitoreo: {e}")
    
    print("\n📝 Prueba completada")