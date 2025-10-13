# 📡 Medición en Tiempo Real

Sistema de adquisición de datos durante competencias para Nissan Sentra V16.

## 🔌 Sensores
- 3x MLX90614 (temperatura superficial de neumáticos)
- IMU del smartphone (aceleraciones y datos de movimiento)
- Célula de carga HX711 (solo para banco de ensayos)

## 📊 Archivos
- `mlx90614_reader.py` - Lectura de sensores de temperatura
- `data_acquisition.py` - Sistema principal de adquisición

## 🚀 Instalación Raspberry Pi
```bash
# Instalar dependencias
sudo pip3 install smbus2 adafruit-circuitpython-mlx90614

# Habilitar I2C
sudo raspi-config nonint do_i2c 0

# Probar sensores
python3 mlx90614_reader.py