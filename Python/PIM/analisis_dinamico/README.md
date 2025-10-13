# 🔬 Análisis Dinámico

Análisis de fuerzas y dinámica vehicular para el Nissan Sentra B16.

## 📊 Archivos Principales
- `tire_analysis_pro.py` - Sistema integrado de análisis
- `config.py` - Parámetros del vehículo

## 🎯 Funcionalidades
- Cálculo de fuerzas por rueda
- Transferencia de carga
- Modelo lineal de neumáticos
- Identificación puntos críticos

## 🚀 Uso
```python
from tire_analysis_pro import TireAnalysisSystem
sistema = TireAnalysisSystem()
resultados = sistema.analyze_track_data(datos_carrera)