"""
Sistema principal de adquisición de datos en tiempo real
"""
import time
import csv
from datetime import datetime
from mlx90614_reader import MLX90614

class DataAcquisitionSystem:
    def __init__(self):
        self.sensores_temperatura = {
            'delantero_izq': MLX90614(address=0x5A),
            'delantero_der': MLX90614(address=0x5B), 
            'trasero_der': MLX90614(address=0x5C)
        }
        self.archivo_datos = None
        self.csv_writer = None
        
    def iniciar_adquisicion(self, nombre_archivo=None):
        if not nombre_archivo:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            nombre_archivo = f"datos_carrera_{timestamp}.csv"
            
        self.archivo_datos = open(nombre_archivo, 'w', newline='')
        self.csv_writer = csv.writer(self.archivo_datos)
        
        # Escribir encabezado
        encabezado = ['timestamp', 'temp_del_izq', 'temp_del_der', 'temp_tras_der']
        self.csv_writer.writerow(encabezado)
        print(f"🎯 Iniciando adquisición: {nombre_archivo}")
        
    def leer_sensores(self):
        datos = {
            'timestamp': datetime.now().isoformat(),
            'temp_del_izq': None,
            'temp_del_der': None, 
            'temp_tras_der': None
        }
        
        for nombre, sensor in self.sensores_temperatura.items():
            temp = sensor.read_temperature()
            if temp:
                clave = f"temp_{nombre}"
                datos[clave] = temp
                
        return datos
    
    def guardar_datos(self, datos):
        if self.csv_writer:
            fila = [
                datos['timestamp'],
                datos['temp_del_izq'] or '',
                datos['temp_del_der'] or '',
                datos['temp_tras_der'] or ''
            ]
            self.csv_writer.writerow(fila)
            self.archivo_datos.flush()
            
    def ejecutar(self, duracion_minutos=60):
        self.iniciar_adquisicion()
        inicio = time.time()
        duracion_segundos = duracion_minutos * 60
        
        try:
            while time.time() - inicio < duracion_segundos:
                datos = self.leer_sensores()
                self.guardar_datos(datos)
                
                # Mostrar en consola
                print(f"T: {datos['timestamp'][11:19]} | "
                      f"DI: {datos['temp_del_izq'] or '---'}°C | "
                      f"DD: {datos['temp_del_der'] or '---'}°C | "
                      f"TD: {datos['temp_tras_der'] or '---'}°C")
                      
                time.sleep(1)  # Muestreo cada 1 segundo
                
        except KeyboardInterrupt:
            print("\n🛑 Adquisición interrumpida por usuario")
        finally:
            if self.archivo_datos:
                self.archivo_datos.close()
            print("✅ Adquisición finalizada")

if __name__ == "__main__":
    sistema = DataAcquisitionSystem()
    print("Sistema de Adquisición - Nissan Sentra B16")
    print("Presiona Ctrl+C para detener")
    sistema.ejecutar(duracion_minutos=120)  # 2 horas por defecto