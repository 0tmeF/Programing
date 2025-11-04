#!/usr/bin/env python3
import smbus
import time
from mlx90614 import MLX90614

print("🌡️  SENSOR MLX90614 - TEMPERATURA INFRARROJA")
print("=" * 45)

try:
    # Inicializar bus I2C y sensor
    bus = smbus.SMBus(1)
    sensor = MLX90614(bus, address=0x5A)
    
    print("✅ Sensor MLX90614 detectado")
    print("📊 Iniciando lecturas... (Ctrl+C para detener)")
    print("-" * 45)
    
    while True:
        try:
            # Leer temperaturas
            temp_objeto = sensor.get_obj_temp()  # Temperatura del objeto apuntado
            temp_ambiente = sensor.get_amb_temp()  # Temperatura ambiente del sensor
            
            print(f"Objeto: {temp_objeto:.2f}°C | Ambiente: {temp_ambiente:.2f}°C", end='\r')
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Error en lectura: {e}")
            time.sleep(2)
            
except KeyboardInterrupt:
    print("\n\n✅ Programa terminado")
except Exception as e:
    print(f"❌ Error inicializando sensor: {e}")
    print("💡 Verifica:")
    print("   - Conexiones I2C (SDA/SCL)")
    print("   - Dirección I2C (sudo i2cdetect -y 1)")
finally:
    try:
        bus.close()
    except:
        pass