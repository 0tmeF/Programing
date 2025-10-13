# -*- coding: utf-8 -*-
"""
Sistema Principal de Adquisición de Datos en Tiempo Real
Nissan Sentra B16 - Carreras Cabrero
"""
import time
import csv
import logging
import json
from datetime import datetime
from pathlib import Path

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataAcquisitionSystem:
    """
    Sistema completo de adquisición de datos para mediciones en pista
    """
    
    def __init__(self, data_directory="datos_carrera"):
        self.data_dir = Path(data_directory)
        self.data_dir.mkdir(exist_ok=True)
        
        self.sensors = {}
        self.data_file = None
        self.csv_writer = None
        self.is_running = False
        self.sample_count = 0
        
        # Metadatos de la sesión
        self.session_metadata = {
            'vehicle': 'Nissan Sentra B16',
            'created_at': datetime.now().isoformat(),
            'sensor_config': {}
        }
        
        logger.info("🚗 Sistema de adquisición inicializado")
    
    def setup_sensors(self):
        """
        Configura todos los sensores del sistema
        """
        try:
            from mlx90614_reader import setup_tire_sensors
            
            print("🔧 CONFIGURANDO SENSORES...")
            self.sensors = setup_tire_sensors()
            
            # Guardar configuración en metadatos
            for name, sensor in self.sensors.items():
                self.session_metadata['sensor_config'][name] = {
                    'address': f"0x{sensor.address:02X}",
                    'connected': sensor.is_connected
                }
            
            logger.info(f"✅ {len(self.sensors)} sensores configurados")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error configurando sensores: {e}")
            return False
    
    def start_session(self, session_name=None, session_notes=""):
        """
        Inicia una nueva sesión de adquisición
        """
        if not session_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"carrera_{timestamp}"
        
        self.session_metadata.update({
            'session_name': session_name,
            'session_notes': session_notes,
            'start_time': datetime.now().isoformat()
        })
        
        # Crear archivo CSV
        filename = self.data_dir / f"{session_name}.csv"
        self.data_file = open(filename, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.data_file)
        
        # Escribir encabezado
        header = [
            'timestamp', 'sample_count',
            'temp_del_izq', 'temp_del_der', 'temp_tras_der'
        ]
        self.csv_writer.writerow(header)
        
        # Guardar metadatos en archivo JSON
        metadata_file = self.data_dir / f"{session_name}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.session_metadata, f, indent=2, ensure_ascii=False)
        
        self.is_running = True
        self.sample_count = 0
        
        logger.info(f"🎯 Sesión iniciada: {session_name}")
        print(f"📁 Datos guardados en: {filename}")
        
        return True
    
    def read_sensor_data(self):
        """
        Lee datos de todos los sensores
        """
        timestamp = datetime.now().isoformat()
        data = {
            'timestamp': timestamp,
            'sample_count': self.sample_count
        }
        
        # Leer sensores de temperatura
        for name, sensor in self.sensors.items():
            # Mapear nombres completos a claves cortas para CSV
            if name == "Delantero Izquierdo":
                key = 'temp_del_izq'
            elif name == "Delantero Derecho":
                key = 'temp_del_der'
            elif name == "Trasero Derecho":
                key = 'temp_tras_der'
            else:
                continue
            
            temperature = sensor.read_temperature()
            data[key] = temperature
        
        self.sample_count += 1
        return data
    
    def save_data(self, data):
        """
        Guarda datos en el archivo CSV
        """
        if self.csv_writer:
            row = [
                data['timestamp'],
                data['sample_count'],
                data.get('temp_del_izq', ''),
                data.get('temp_del_der', ''),
                data.get('temp_tras_der', '')
            ]
            self.csv_writer.writerow(row)
            self.data_file.flush()  # Forzar escritura inmediata
    
    def display_data(self, data):
        """
        Muestra datos en formato legible en consola
        """
        timestamp = data['timestamp'][11:19]  # HH:MM:SS
        sample = data['sample_count']
        
        temp_di = data.get('temp_del_izq', '---')
        temp_dd = data.get('temp_del_der', '---')
        temp_td = data.get('temp_tras_der', '---')
        
        print(f"#{sample:4d} 🕒 {timestamp} | "
              f"🚗 DI: {str(temp_di):>5} | "
              f"🚗 DD: {str(temp_dd):>5} | "
              f"🚗 TD: {str(temp_td):>5}")
    
    def stop_session(self):
        """
        Detiene la sesión de adquisición
        """
        self.is_running = False
        
        if self.data_file:
            self.data_file.close()
        
        self.session_metadata['end_time'] = datetime.now().isoformat()
        self.session_metadata['total_samples'] = self.sample_count
        
        logger.info(f"🛑 Sesión finalizada. Muestras: {self.sample_count}")
        print(f"✅ Adquisición completada - {self.sample_count} muestras")
    
    def run(self, duration_min=120, sample_interval=1):
        """
        Ejecuta el sistema de adquisición principal
        """
        print("🚗 SISTEMA DE ADQUISICIÓN - NISSAN SENTRA B16")
        print("=" * 60)
        
        # Configurar sensores
        if not self.setup_sensors():
            print("❌ Error crítico: No se pudieron configurar los sensores")
            return
        
        # Iniciar sesión
        session_name = input("Ingrese nombre de la sesión (Enter para automático): ").strip()
        session_notes = input("Notas de la sesión (opcional): ").strip()
        
        if not self.start_session(session_name or None, session_notes):
            print("❌ Error iniciando sesión")
            return
        
        print(f"\n⏱️  Duración programada: {duration_min} minutos")
        print(f"📊 Intervalo de muestreo: {sample_interval} segundo(s)")
        print("🎯 Presione Ctrl+C para detener la adquisición")
        print("-" * 60)
        
        start_time = time.time()
        duration_sec = duration_min * 60
        
        try:
            while self.is_running and (time.time() - start_time < duration_sec):
                # Leer datos
                data = self.read_sensor_data()
                
                # Guardar datos
                self.save_data(data)
                
                # Mostrar en consola
                self.display_data(data)
                
                # Esperar hasta siguiente muestra
                time.sleep(sample_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Adquisición interrumpida por usuario")
        except Exception as e:
            logger.error(f"Error durante adquisición: {e}")
            print(f"❌ Error: {e}")
        finally:
            self.stop_session()
            
            # Mostrar resumen
            elapsed = time.time() - start_time
            samples_per_sec = self.sample_count / elapsed if elapsed > 0 else 0
            print(f"\n📈 RESUMEN FINAL:")
            print(f"   Tiempo total: {elapsed:.1f} segundos")
            print(f"   Muestras: {self.sample_count}")
            print(f"   Frecuencia: {samples_per_sec:.1f} Hz")

# Función principal
if __name__ == "__main__":
    # Crear sistema
    acquisition_system = DataAcquisitionSystem()
    
    # Configurar parámetros
    try:
        duration = int(input("Duración en minutos [120]: ") or "120")
        interval = float(input("Intervalo de muestreo en segundos [1.0]: ") or "1.0")
    except ValueError:
        print("⚠️  Usando valores por defecto")
        duration = 120
        interval = 1.0
    
    # Ejecutar sistema
    acquisition_system.run(duration_min=duration, sample_interval=interval)