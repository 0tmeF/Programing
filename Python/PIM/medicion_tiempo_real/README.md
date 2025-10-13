# 📡 Medición en Tiempo Real

Sistema de adquisición de datos durante competencias.

## 🔌 Sensores
- 3x MLX90614 (temperatura superficial)
- IMU del smartphone (aceleraciones)
- Célula de carga HX711 (banco ensayos)

## 📊 Por Implementar
- `mlx90614_reader.py` - Lectura sensores temperatura
- `data_acquisition.py` - Sistema principal adquisición
- `imu_data_sync.py` - Sincronización con datos IMU

## 🚀 Instalación Raspberry Pi
```bash
sudo pip3 install smbus2 adafruit-circuitpython-mlx90614
sudo raspi-config nonint do_i2c 0