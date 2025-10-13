# ⚖️ Banco de Ensayos

Caracterización estática de coeficiente de fricción vs temperatura.

## 🔧 Equipamiento
- Célula de carga HX711
- Sensores MLX90614
- Raspberry Pi 3

## 📊 Archivos Principales  
- `friction_calibration.py` - Calibración principal
- `hx711_interface.py` - Interface célula carga

## 🧪 Protocolo
1. Estabilizar temperatura
2. Aplicar carga normal
3. Medir fuerza arrastre
4. Calcular μ = F_arrastre / F_normal