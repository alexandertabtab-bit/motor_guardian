/*
  MOTOR GUARDIAN - Arduino Uno hardware firmware v1
  Hardware:
    MPU6050: SDA=A4, SCL=A5, VCC=5V or 3.3V as required by breakout, GND=GND
    IR obstacle sensor OUT=D2, VCC/GND as required by module
    Relay IN=D7 (relay contacts switch LOW-VOLTAGE fan supply only)
    Buzzer=D8, green LED=D9, red LED=D10 (LEDs need series resistors)

  Serial protocol at 115200 baud:
    PC -> PING
    PC -> RELAY,1
    PC -> RELAY,0
    Arduino -> DATA,millis,vib_g,chip_temp_c,rpm,relay,pulses,sensor_ok

  SAFETY:
    - Relay is OFF on boot, MPU6050 failure, or loss of PC heartbeat for 3 seconds.
    - Use relay contacts ONLY with the low-voltage demo fan. Never connect mains.
    - Verify whether your relay module is active-low. Change RELAY_ACTIVE_LOW if needed.
    - IR sensor must see one reflective mark/slot per revolution for PULSES_PER_REV=1.
*/

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

const uint8_t IR_PIN = 2;       // Uno external interrupt pin
const uint8_t RELAY_PIN = 7;
const uint8_t BUZZER_PIN = 8;
const uint8_t GREEN_LED_PIN = 9;
const uint8_t RED_LED_PIN = 10;

const bool RELAY_ACTIVE_LOW = true;  // Common relay-module setting; verify your board.
const uint8_t PULSES_PER_REV = 1;    // One reflective mark/slot per fan revolution.
const unsigned long SAMPLE_INTERVAL_MS = 20;   // ~50 Hz accelerometer sampling
const unsigned long REPORT_INTERVAL_MS = 250;  // 4 telemetry reports per second
const unsigned long HEARTBEAT_TIMEOUT_MS = 3000;

volatile unsigned long irPulses = 0;
unsigned long lastHeartbeatMs = 0;
unsigned long lastSampleMs = 0;
unsigned long lastReportMs = 0;
unsigned long lastPulseSnapshot = 0;

float vibrationSumSq = 0.0f;
unsigned int vibrationSamples = 0;
bool mpuOk = false;
bool relayOn = false;
float latestChipTempC = NAN;
float latestVibrationG = 0.0f;
float latestRpm = 0.0f;

void countIrPulse() {
  irPulses++;
}

void setRelay(bool on) {
  relayOn = on;
  const uint8_t energized = RELAY_ACTIVE_LOW ? LOW : HIGH;
  const uint8_t deenergized = RELAY_ACTIVE_LOW ? HIGH : LOW;
  digitalWrite(RELAY_PIN, on ? energized : deenergized);
  digitalWrite(GREEN_LED_PIN, on ? HIGH : LOW);
}

void safeStop() {
  setRelay(false);
  digitalWrite(BUZZER_PIN, HIGH);
  digitalWrite(RED_LED_PIN, HIGH);
}

void clearAlarmIndicators() {
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(RED_LED_PIN, LOW);
}

void handleCommand(String command) {
  command.trim();
  if (command == "PING") {
    lastHeartbeatMs = millis();
    return;
  }

  if (command == "RELAY,0") {
    setRelay(false);
    clearAlarmIndicators();
    return;
  }

  if (command == "RELAY,1") {
    // Require a recent PC heartbeat and a working accelerometer.
    if (mpuOk && (millis() - lastHeartbeatMs <= HEARTBEAT_TIMEOUT_MS)) {
      setRelay(true);
      clearAlarmIndicators();
    } else {
      safeStop();
    }
  }
}

void readSerialCommands() {
  static String input = "";
  while (Serial.available() > 0) {
    char ch = (char)Serial.read();
    if (ch == '\n') {
      handleCommand(input);
      input = "";
    } else if (ch != '\r' && input.length() < 40) {
      input += ch;
    } else if (input.length() >= 40) {
      input = "";
    }
  }
}

void setup() {
  // Set output pins to safe state immediately.
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(GREEN_LED_PIN, OUTPUT);
  pinMode(RED_LED_PIN, OUTPUT);
  setRelay(false);
  clearAlarmIndicators();

  pinMode(IR_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(IR_PIN), countIrPulse, FALLING);

  Serial.begin(115200);
  Wire.begin();

  mpuOk = mpu.begin();
  if (mpuOk) {
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  } else {
    safeStop();
  }

  lastHeartbeatMs = millis();
  lastSampleMs = millis();
  lastReportMs = millis();
}

void loop() {
  readSerialCommands();

  const unsigned long now = millis();

  // Fail-safe: stop if the dashboard/PC stops sending heartbeat messages.
  if (now - lastHeartbeatMs > HEARTBEAT_TIMEOUT_MS) {
    safeStop();
  }

  if (!mpuOk) {
    safeStop();
  }

  // Accelerometer sampling at approximately 50 Hz.
  if (mpuOk && now - lastSampleMs >= SAMPLE_INTERVAL_MS) {
    lastSampleMs = now;
    sensors_event_t accel, gyro, temp;
    mpu.getEvent(&accel, &gyro, &temp);

    const float magnitude = sqrt(
      accel.acceleration.x * accel.acceleration.x +
      accel.acceleration.y * accel.acceleration.y +
      accel.acceleration.z * accel.acceleration.z
    );

    // Approximate dynamic acceleration magnitude after removing gravity.
    // This is acceleration in g, NOT vibration velocity in mm/s.
    const float dynamicG = (magnitude - 9.80665f) / 9.80665f;
    vibrationSumSq += dynamicG * dynamicG;
    vibrationSamples++;
    latestChipTempC = temp.temperature;
  }

  // Report telemetry four times per second.
  if (now - lastReportMs >= REPORT_INTERVAL_MS) {
    const unsigned long elapsedMs = now - lastReportMs;
    lastReportMs = now;

    if (vibrationSamples > 0) {
      latestVibrationG = sqrt(vibrationSumSq / vibrationSamples);
    } else {
      latestVibrationG = 0.0f;
    }
    vibrationSumSq = 0.0f;
    vibrationSamples = 0;

    unsigned long pulseSnapshot;
    noInterrupts();
    pulseSnapshot = irPulses;
    interrupts();

    const unsigned long newPulses = pulseSnapshot - lastPulseSnapshot;
    lastPulseSnapshot = pulseSnapshot;
    latestRpm = (elapsedMs > 0)
      ? (newPulses * 60000.0f) / (PULSES_PER_REV * elapsedMs)
      : 0.0f;

    Serial.print("DATA,");
    Serial.print(now);
    Serial.print(",");
    Serial.print(latestVibrationG, 5);
    Serial.print(",");
    Serial.print(latestChipTempC, 2);
    Serial.print(",");
    Serial.print(latestRpm, 1);
    Serial.print(",");
    Serial.print(relayOn ? 1 : 0);
    Serial.print(",");
    Serial.print(newPulses);
    Serial.print(",");
    Serial.println(mpuOk ? 1 : 0);
  }
}
