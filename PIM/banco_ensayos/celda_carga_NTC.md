#include "HX711.h"

// ================= CONFIGURACIÓN PINES =================
const int LOADCELL_DOUT_PIN = A1;
const int LOADCELL_SCK_PIN = A0;
const int NTC_PIN = A2;
const int NTC_SERIES_RESISTOR = 100000;

// ================= PARÁMETROS NTC AJUSTABLES =================
float NTC_NOMINAL = 100000.0;      // Ajustable durante calibración
float TEMP_NOMINAL = 25.0;
int B_COEFFICIENT = 3950;
float temp_offset = 0.0;           // Corrección de temperatura

// ================= VARIABLES =================
HX711 scale;
float calibration_factor = 1000.0;

// ================= SETUP =================
void setup() {
  Serial.begin(9600);
  Serial.println("🎯 SISTEMA COMPLETO - CALIBRACIÓN TEMP + PESO");
  Serial.println("=============================================");
  
  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  scale.set_scale(calibration_factor);
  scale.tare();
  
  Serial.println("✅ Sistema listo");
  Serial.println("📋 COMANDOS:");
  Serial.println("  t - Tare (poner a cero balanza)");
  Serial.println("  c - Calibrar peso con peso conocido"); 
  Serial.println("  T - Calibrar temperatura con temp conocida");
  Serial.println("  o - Ajustar offset temperatura (+/-0.5°C)");
  Serial.println("  i - Mostrar información de calibración");
  Serial.println("=============================================");
}

// ================= LECTURA NTC =================
float leerTemperaturaNTC() {
  int lecturaADC = analogRead(NTC_PIN);
  
  Serial.print(" [ADC:");
  Serial.print(lecturaADC);
  
  double resistenciaNTC = 100000.0 / ((1023.0 / lecturaADC) - 1);
  
  Serial.print(" R:");
  Serial.print(resistenciaNTC / 1000.0, 1);
  Serial.print("k] ");
  
  if (resistenciaNTC < 1000.0 || resistenciaNTC > 500000.0 || lecturaADC <= 10 || lecturaADC >= 1013) {
    return -999.0;
  }
  
  double steinhart;
  steinhart = resistenciaNTC / NTC_NOMINAL;
  steinhart = log(steinhart);
  steinhart /= B_COEFFICIENT;
  steinhart += 1.0 / (TEMP_NOMINAL + 273.15);
  steinhart = 1.0 / steinhart;
  steinhart -= 273.15;
  
  // Aplicar offset de calibración
  return steinhart + temp_offset;
}

// ================= CALIBRACIÓN PESO =================
void calibrarPeso() {
  Serial.println();
  Serial.println("🎯 CALIBRACIÓN PESO - Coloca peso conocido (gramos):");
  
  while (!Serial.available()) delay(100);
  float pesoConocido = Serial.parseFloat();
  
  if (pesoConocido <= 0) {
    Serial.println("❌ Peso debe ser mayor a 0");
    return;
  }
  
  Serial.print("🔧 Calibrando con ");
  Serial.print(pesoConocido);
  Serial.println("g...");
  
  long rawValue = scale.get_value(10);
  calibration_factor = rawValue / pesoConocido;
  scale.set_scale(calibration_factor);
  
  Serial.print("✅ Nuevo factor peso: ");
  Serial.println(calibration_factor, 1);
  
  // Verificación
  float pesoVerificado = scale.get_units(5);
  Serial.print("📊 Peso verificado: ");
  Serial.print(pesoVerificado, 1);
  Serial.println("g");
}

// ================= CALIBRACIÓN TEMPERATURA =================
void calibrarTemperatura() {
  Serial.println();
  Serial.println("🎯 CALIBRACIÓN TEMPERATURA");
  Serial.println("Conoce la temperatura actual (ej: 25.0):");
  
  // Leer temperatura actual
  float tempActual = leerTemperaturaSinOffset();
  Serial.print("🌡️  Temperatura medida: ");
  Serial.print(tempActual, 1);
  Serial.println("°C");
  
  Serial.println("Ingresa temperatura REAL conocida (°C):");
  
  while (!Serial.available()) delay(100);
  float tempReal = Serial.parseFloat();
  
  // Calcular offset
  temp_offset = tempReal - tempActual;
  
  Serial.print("✅ Offset aplicado: ");
  Serial.print(temp_offset, 1);
  Serial.println("°C");
  
  Serial.print("🎯 Nueva temperatura: ");
  Serial.print(tempActual + temp_offset, 1);
  Serial.println("°C");
}

float leerTemperaturaSinOffset() {
  // Lectura sin offset para calibración
  int lecturaADC = analogRead(NTC_PIN);
  double resistenciaNTC = 100000.0 / ((1023.0 / lecturaADC) - 1);
  
  if (resistenciaNTC < 1000.0 || resistenciaNTC > 500000.0) {
    return -999.0;
  }
  
  double steinhart;
  steinhart = resistenciaNTC / NTC_NOMINAL;
  steinhart = log(steinhart);
  steinhart /= B_COEFFICIENT;
  steinhart += 1.0 / (TEMP_NOMINAL + 273.15);
  steinhart = 1.0 / steinhart;
  steinhart -= 273.15;
  
  return steinhart;
}

// ================= AJUSTE OFFSET TEMPERATURA =================
void ajustarOffsetTemperatura() {
  Serial.println();
  Serial.println("🔧 AJUSTAR OFFSET TEMPERATURA");
  Serial.println("+ - Aumentar 0.5°C");
  Serial.println("- - Disminuir 0.5°C");
  Serial.println("0 - Resetear offset");
  Serial.print("Offset actual: ");
  Serial.print(temp_offset, 1);
  Serial.println("°C");
  
  while (!Serial.available()) delay(100);
  char ajuste = Serial.read();
  
  switch (ajuste) {
    case '+':
      temp_offset += 0.5;
      break;
    case '-':
      temp_offset -= 0.5;
      break;
    case '0':
      temp_offset = 0.0;
      break;
    default:
      Serial.println("❌ Comando no válido");
      return;
  }
  
  Serial.print("✅ Nuevo offset: ");
  Serial.print(temp_offset, 1);
  Serial.println("°C");
}

// ================= INFORMACIÓN CALIBRACIÓN =================
void mostrarInformacionCalibracion() {
  Serial.println();
  Serial.println("📊 INFORMACIÓN DE CALIBRACIÓN");
  Serial.println("=============================");
  Serial.print("🔸 Factor peso: ");
  Serial.println(calibration_factor, 1);
  Serial.print("🔸 Offset temperatura: ");
  Serial.print(temp_offset, 1);
  Serial.println("°C");
  Serial.print("🔸 NTC Nominal: ");
  Serial.print(NTC_NOMINAL / 1000, 0);
  Serial.println("kΩ");
  Serial.print("🔸 Coeficiente B: ");
  Serial.println(B_COEFFICIENT);
  Serial.println("=============================");
}

// ================= LOOP PRINCIPAL =================
void loop() {
  // Leer sensores
  float peso = scale.get_units(3);
  float temperatura = leerTemperaturaNTC();
  
  // Mostrar resultados
  Serial.print("⚖️  Peso: ");
  Serial.print(peso, 1);
  Serial.print("g | 🌡️  Temp: ");
  
  if (temperatura < -100.0) {
    Serial.print("ERROR");
  } else {
    Serial.print(temperatura, 1);
    Serial.print("°C");
  }
  
  // Procesar comandos
  if (Serial.available()) {
    char comando = Serial.read();
    switch (comando) {
      case 't':  // Tare
        scale.tare();
        Serial.println();
        Serial.println("✅ Tare realizado - Peso en CERO");
        break;
      case 'c':  // Calibrar peso
        calibrarPeso();
        break;
      case 'T':  // Calibrar temperatura
        calibrarTemperatura();
        break;
      case 'o':  // Ajustar offset temperatura
        ajustarOffsetTemperatura();
        break;
      case 'i':  // Información
        mostrarInformacionCalibracion();
        break;
    }
    while (Serial.available()) Serial.read();
  } else {
    Serial.println();
  }
  
  delay(2000);
}