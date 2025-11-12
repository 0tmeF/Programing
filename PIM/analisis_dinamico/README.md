# 🔬 Análisis Dinámico

Análisis de fuerzas, transferencia de carga y dinámica vehicular para el Nissan Sentra V16.

## 📊 Archivos Principales
- `tire_dynamics_v16.py` - Sistema integrado de análisis
- `config.py` - Parámetros del vehículo

## 🎯 Funcionalidades
- Cálculo de fuerzas normales por rueda
- Transferencia de carga longitudinal y lateral
- Modelo lineal de neumáticos
- Identificación de puntos críticos

## 🚀 Uso
```python
from tire_dynamics_v16 import TireAnalysisSystem
sistema = TireAnalysisSystem()
resultados = sistema.analyze_track_data(datos_carrera)