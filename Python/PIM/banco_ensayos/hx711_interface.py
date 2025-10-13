"""
Interface para célula de carga HX711 en banco de ensayos
"""
import time
import RPi.GPIO as GPIO

class HX711:
    def __init__(self, dout_pd_sck, gain=128, gpio_mode=None):
        self.DOUT = dout_pd_sck[0]
        self.PD_SCK = dout_pd_sck[1]
        
        if gpio_mode is not None:
            GPIO.setmode(gpio_mode)
        GPIO.setup(self.DOUT, GPIO.IN)
        GPIO.setup(self.PD_SCK, GPIO.OUT)
        
        self.GAIN = gain
        self.GAIN_PULSES = {128: 1, 64: 3, 32: 2}.get(gain, 1)
        self.OFFSET = 0
        self.SCALE = 1
        
        self.set_gain()
        
    def set_gain(self):
        GPIO.output(self.PD_SCK, False)
        self.read()
        
    def read(self):
        # Esperar hasta que DOUT esté listo
        while GPIO.input(self.DOUT) != 0:
            time.sleep(0.001)
            
        data = 0
        for _ in range(24):
            GPIO.output(self.PD_SCK, True)
            data <<= 1
            GPIO.output(self.PD_SCK, False)
            if GPIO.input(self.DOUT):
                data |= 1
        # Configurar ganancia
        gain_pulses = {128: 1, 64: 2, 32: 3}.get(self.GAIN, 1)
        for _ in range(gain_pulses):
            GPIO.output(self.PD_SCK, True)
            GPIO.output(self.PD_SCK, False)
            GPIO.output(self.PD_SCK, False)
        # Convertir a signed integer
        if data & 0x800000:
            data -= 0x1000000

        return data
        return data
    def get_weight(self, samples=10):
        """
        Obtiene el peso promedio en Newtons.

        Args:
            samples (int): Número de lecturas a promediar.

        Returns:
            float: Fuerza promedio en Newtons calculada a partir de 'samples' lecturas.
        """
        total = 0
        for _ in range(samples):
            total += self.read() - self.OFFSET
            time.sleep(0.01)
            
        avg = total / samples
        weight_newtons = avg / self.SCALE
        return weight_newtons
        return weight_newtons

# Ejemplo de calibración
if __name__ == "__main__":
    # Configurar pines (ajustar según conexión)
    hx = HX711((5, 6))  # DOUT=GPIO5, PD_SCK=GPIO6
    
    print("Calibración HX711 - Banco de Ensayos")
    print("1. Sin carga - presiona Enter para calibrar offset")
    input()
    hx.OFFSET = hx.read()
    print(f"Offset calibrado: {hx.OFFSET}")
    
    print("2. Coloca peso conocido (ej: 10kg = 98.1N) y presiona Enter")
    input()
    raw_value = hx.read() - hx.OFFSET
    known_weight = 98.1  # 10kg en Newtons
    if known_weight == 0:
        print("Error: El peso conocido no puede ser cero para la calibración de escala.")
        exit(1)
    hx.SCALE = raw_value / known_weight
    print(f"Escala calibrada: {hx.SCALE}")
    
    print("Listo para mediciones...")
    while True:
        weight = hx.get_weight()
        print(f"Fuerza: {weight:.2f} N")
        time.sleep(1)