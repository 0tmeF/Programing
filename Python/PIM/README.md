# 🏎️ PIM - Optimización Nissan Sentra V16

Sistema de medición y análisis para optimizar performance en carreras de Cabrero.

## 📋 Proyecto
- **Vehículo**: Nissan Sentra V16 (2001) - Motor GA16DNE/GA16DE  
- **Categoría**: V16 Estándar - Carreras Cabrero, Biobío
- **Objetivo**: Optimizar performance mediante análisis térmico de neumáticos

## 🔬 Sistema de Medición
- **Temperatura**: 3x sensores MLX90614
- **Fuerzas**: Célda carga HX711 (banco ensayos)
- **Control**: Raspberry Pi 3 + Python 3

## 📁 Estructura
PIM/
-> analisis_dinamico/ # Análisis fuerzas y dinámica
-> banco_ensayos/ # Calibración y caracterización
-> medicion_tiempo_real/ # Adquisición datos en pista
-> datos/ # Datos experimentales
-> docs/ # Documentación
-> config/ # Configuraciones


## 🚀 Uso Rápido
```bash
# Análisis dinámico
cd analisis_dinamico
python tire_analysis_pro.py

# Banco de ensayos  
cd banco_ensayos
python friction_calibration.py

Equipo: Carlos Caamaño, Jesus Duarte, Hugo Zambrano
Fecha: Octubre 2025