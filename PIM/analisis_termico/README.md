# 🌡️ Análisis Térmico

Análisis de temperatura de neumáticos y su relación con el coeficiente de fricción para el Nissan Sentra B16.

## 🎯 Objetivo
Correlacionar la temperatura de los neumáticos con el coeficiente de fricción para optimizar el performance en pista.

## 🔬 Metodología

### 1. **Medición en Tiempo Real**
- 3 sensores MLX90614 apuntando a neumáticos
- Adquisición continua durante carreras
- Sincronización con datos de aceleración

### 2. **Caracterización en Banco de Ensayos**
- Curva μ vs Temperatura para neumáticos semi-slick
- Validación de coeficientes de fricción
- Calibración de modelos térmicos

### 3. **Modelado Térmico**
- Transferencia de calor en neumáticos
- Efecto de la presión sobre temperatura
- Modelos predictivos de temperatura óptima

## 📊 Parámetros Clave

| Parámetro | Valor Óptimo | Rango Operativo |
|-----------|-------------|-----------------|
| Temp. trabajo | 70-90°C | 50-110°C |
| μ máximo | ~1.2 | 0.8-1.4 |
| Calentamiento | 2-3 vueltas | - |

## 🔧 Archivos Principales
- `modelo_termico.py` - Modelos de transferencia de calor
- `correlacion_temperatura_friccion.py` - Análisis μ vs T
- `optimizacion_setup.py` - Recomendaciones de setup

## 📈 Análisis Planificados

### Corto Plazo
- [ ] Curva μ vs T para neumáticos actuales
- [ ] Modelo de calentamiento por vuelta
- [ ] Identificación de sobrecalentamiento

### Medio Plazo  
- [ ] Modelo predictivo de temperatura
- [ ] Optimización de presiones por temperatura
- [ ] Análisis de degradación térmica

## 🚀 Uso Rápido
```python
from modelo_termico import AnalisisTermico

analisis = AnalisisTermico()
temperaturas = analisis.cargar_datos('datos/carrera_ultima.csv')
resultados = analisis.correlacionar_temperatura_friccion(temperaturas)
analisis.generar_recomendaciones()