# Motor Guardian — hardware prototype v1

This version connects an **Arduino Uno** to a **Streamlit dashboard over USB serial**.

## Hardware used

- Arduino Uno
- MPU6050 accelerometer/gyroscope module
- IR obstacle sensor (used as a pulse sensor for RPM)
- Relay module
- Low-voltage demo fan
- Buzzer and two LEDs (optional indicators)

## Important measurement limits

This hardware measures:
- Vibration as an acceleration-based RMS-like value in **g**
- MPU6050 internal chip temperature (not motor surface temperature)
- RPM, if the IR sensor detects one marker per revolution

It does **not** physically measure:
- Three-phase current or phase imbalance
- Insulation resistance
- Acoustic noise
- Motor surface temperature

Those signals must not be presented as live physical measurements.

## Wiring

| Module | Arduino Uno |
|---|---|
| MPU6050 SDA | A4 |
| MPU6050 SCL | A5 |
| MPU6050 GND | GND |
| MPU6050 VCC | Module-rated supply (check breakout board) |
| IR sensor OUT | D2 |
| IR sensor GND | GND |
| IR sensor VCC | Module-rated supply |
| Relay IN | D7 |
| Buzzer signal | D8 |
| Green LED (with resistor) | D9 |
| Red LED (with resistor) | D10 |

Connect grounds as appropriate for the low-voltage control circuit. Check the pin labels on your exact modules before powering them.

### Fan and relay safety

**Only switch the low-voltage demo fan. Never connect household mains or a factory motor to this prototype relay.** Have the wiring checked by a qualified supervisor. Keep the fan mechanically guarded and disconnect power before changing wiring.

The firmware assumes a common active-LOW relay module. If your module is active-HIGH, change `RELAY_ACTIVE_LOW` in the sketch. Test relay behavior with the fan supply disconnected first.

## Arduino libraries

Install these through Arduino IDE Library Manager:
1. Adafruit MPU6050
2. Adafruit Unified Sensor
3. Adafruit BusIO (usually installed as a dependency)

Open `motor_guardian.ino`, select **Arduino Uno** and the correct port, then upload.

## Python setup (Windows)

Open Command Prompt in this folder:

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Select the Arduino COM port in the sidebar. Close Arduino IDE Serial Monitor before connecting.

## Serial protocol

The Arduino sends:

```text
DATA,millis,vib_g,chip_temp_c,rpm,relay,pulses,sensor_ok
```

The dashboard sends:
- `PING` — heartbeat
- `RELAY,1` — request relay ON
- `RELAY,0` — request relay OFF

The Arduino switches the relay OFF after 3 seconds without a heartbeat or if the MPU6050 fails to initialize.

## First test sequence

1. Upload the firmware with the relay/fan power disconnected.
2. Open Serial Monitor at 115200 baud and confirm `DATA,...` lines appear.
3. Close Serial Monitor.
4. Run the Streamlit dashboard and connect to the Arduino port.
5. Verify sensor values change when you gently move the MPU6050.
6. Verify RPM changes when the IR sensor sees the fan marker.
7. Test relay ON/OFF with the low-voltage fan only, under supervision.
8. Keep the fan in a known healthy condition for the first 60 telemetry samples so the dashboard can learn its baseline.

## Current scope

This is the first real-hardware milestone: sensor acquisition, USB telemetry, basic vibration anomaly monitoring, operator relay commands, and a communication-loss fail-safe.

It is **not yet** a trained AI fault classifier or a certified industrial protection system. Do not use it to protect real industrial machinery.
