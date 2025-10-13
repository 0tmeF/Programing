# 🌡️ Análisis Térmico - Nissan Sentra V16

Análisis de la relación entre temperatura de neumáticos y performance para optimización del setup en pista.

## 📊 Contenido Actual

### 🔧 Archivos Implementados
- `config.py` - Configuración de parámetros térmicos y del vehículo
- `README.md` - Documentación del módulo

## 🎯 Objetivos del Módulo

### 1. **Caracterización Térmica**
- Establecer curva μ vs Temperatura para neumáticos semi-slick
- Determinar temperaturas óptimas de trabajo (70-90°C)
- Identificar límites térmicos operativos

### 2. **Modelado Predictivo**
- Desarrollar modelos de transferencia de calor
- Predecir evolución de temperatura por vuelta
- Simular efecto de condiciones ambientales

### 3. **Optimización en Pista**
- Generar recomendaciones de setup basadas en temperatura
- Desarrollar estrategias de calentamiento óptimas
- Implementar sistema de alertas térmicas

## 📈 Parámetros de Referencia

|   Parámetro   | Valor Óptimo | Rango Seguro |
|---------------|--------------|--------------|
| Temp. trabajo |    70-90°C   |   50-110°C   |
| μ máximo      |     ~1.2     |   0.8-1.4    |
| Calentamiento | 2-3 vueltas  |      -       |

## 🔧 Archivos por Desarrollar

- `modelo_termico.py` - Modelos de transferencia de calor
- `correlacion_mu_temperatura.py` - Análisis μ vs T  
- `optimizacion_setup.py` - Recomendaciones de setup
- `procesamiento_datos.py` - Herramientas de análisis

## 🚀 Próximos Pasos

### Corto Plazo
- [ ] Implementar adquisición MLX90614
- [ ] Diseñar experimentos en banco de ensayos
- [ ] Desarrollar modelos térmicos básicos

### Medio Plazo
- [ ] Validar con datos reales de carrera
- [ ] Crear herramientas de análisis
- [ ] Desarrollar sistema de recomendaciones

## 🔗 Integración

- **medicion_tiempo_real**: Datos de temperatura en pista
- **banco_ensayos**: Curvas μ(T) experimentales  
- **analisis_dinamico**: Relación fuerzas-generación de calor

---

**Estado**: 🟡 EN DESARROLLO  
**Vehículo**: Nissan Sentra V16 - GA16DE  
**Categoría**: V16 Estándar - Carreras Cabrero