# ⚖️ Banco de Ensayos

Sistema de caracterización estática de coeficiente de fricción vs temperatura para neumáticos semi-slick.

## 🔧 Configuración Experimental
- Célula de carga HX711 (0-50kg)
- Sensores MLX90614 para temperatura
- Raspberry Pi 3 como controlador
- Sistema de calentamiento controlado

## 📊 Archivos
- `banco_ensayos.py` - Sistema principal de calibración
- `hx711_interface.py` - Interface con célula de carga

## 🧪 Protocolo
1. Estabilizar temperatura de probeta
2. Aplicar carga normal conocida
3. Medir fuerza de arrastre
4. Calcular μ = F_arrastre / F_normal
5. Repetir para diferentes temperaturas

## 🎯 Objetivo
Generar curva de coeficiente de fricción vs temperatura para optimización en pista.