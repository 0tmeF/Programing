# 🔬 Análisis Dinámico

Análisis de fuerzas, transferencia de carga y dinámica vehicular para el Nissan Sentra V16.

## 📊 Archivos Principales
- `tire_analysis_pro.py` - Sistema integrado de análisis
- `config.py` - Parámetros del vehículo
- `f_num_2.py` - Versión anterior (referencia)
- `f_neumaticos.py` - Modelo lineal (referencia)

## 🎯 Funcionalidades
- Cálculo de fuerzas normales por rueda
- Transferencia de carga longitudinal y lateral
- Modelo lineal de neumáticos
- Identificación de puntos críticos

## 🚀 Uso
```python
from tire_analysis_pro import TireAnalysisSystem
sistema = TireAnalysisSystem()
resultados = sistema.analyze_track_data(datos_carrera)