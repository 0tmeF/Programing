# ⚖️ Banco de Ensayos

Sistema de caracterización estática de coeficiente de fricción vs temperatura para neumáticos semi-slick.

## 🔧 Configuración Experimental
- Célula de carga HX711 (0-50kg)
- Sensores NTC 100K para la temperatura de la probeta
- Arduino UNO como controlador
- Sistema de control de temperatura

## 📊 Archivos
- `banco_ensayos.py` - Codigo de procesamiento de datos experimentales
- `celda_carga_NTC` - Cadigo de lectura y calibracion de celda de carga y sensor NTC 100k

## 🧪 Protocolo
1. Estabilizar temperatura de probeta
2. Aplicar carga normal conocida
3. Medir fuerza de arrastre en el tiempo: F_arrastre dt
4. Calcular μ dt = (F_arrastre / F_normal) dt
5. Repetir ensayo para diferentes temperaturas

## 🎯 Objetivo
Generar curva de coeficiente de fricción vs temperatura para optimización en pista.