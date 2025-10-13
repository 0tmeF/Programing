"""
Lector de sensores MLX90614 para medición de temperatura en neumáticos
"""
import smbus2
import time

class MLX90614:
    def __init__(self, bus=1, address=0x5A):
        self.bus = smbus2.SMBus(bus)
        self.address = address
        
    def read_temperature(self):
        """Lee temperatura del sensor en Celsius"""
        try:
            # Leer registro de temperatura objeto
            data = self.bus.read_i2c_block_data(self.address, 0x07, 3)
            temp = (data[1] << 8) | data[0]
            temp_celsius = (temp * 0.02) - 273.15
            return round(temp_celsius, 2)
        except Exception as e:
            print(f"Error leyendo sensor: {e}")
            return None

# Ejemplo de uso
if __name__ == "__main__":
    # Crear 3 sensores (diferentes direcciones I2C)
    sensores = {
        'neumatico_delantero_izq': MLX90614(address=0x5A),
        'neumatico_delantero_der': MLX90614(address=0x5B),
        'neumatico_trasero_der': MLX90614(address=0x5C)
    }
    
    while True:
        for nombre, sensor in sensores.items():
            temp = sensor.read_temperature()
            if temp:
                print(f"{nombre}: {temp}°C")
        time.sleep(1)