"""
Motor Guardian - hardware-connected Streamlit dashboard.
Connects to an Arduino Uno over USB serial.
The Arduino firmware is in motor_guardian.ino.
"""
import time
import threading

import numpy as np
import pandas as pd
import serial
from serial.tools import list_ports
import streamlit as st

st.set_page_config(page_title="Motor Guardian", page_icon="⚙️", layout="wide")

BAUD_RATE = 115200
BASELINE_SAMPLES = 60
WARNING_Z = 3.0
CRITICAL_Z = 6.0
PERSISTENCE = 3
MAX_ROWS = 1200

if "serial_conn" not in st.session_state:
    st.session_state.serial_conn = None
if "telemetry" not in st.session_state:
    st.session_state.telemetry = []
if "baseline" not in st.session_state:
    st.session_state.baseline = None
if "high_count" not in st.session_state:
    st.session_state.high_count = 0
if "critical_count" not in st.session_state:
    st.session_state.critical_count = 0
if "auto_protection" not in st.session_state:
    st.session_state.auto_protection = False
if "protection_latched" not in st.session_state:
    st.session_state.protection_latched = False
if "last_error" not in st.session_state:
    st.session_state.last_error = ""


def send_command(command: str) -> bool:
    conn = st.session_state.serial_conn
    if conn is None or not conn.is_open:
        return False
    try:
        conn.write((command + "\n").encode("ascii"))
        return True
    except (serial.SerialException, OSError) as exc:
        st.session_state.last_error = str(exc)
        return False


def close_connection():
    conn = st.session_state.serial_conn
    if conn is not None:
        try:
            conn.write(b"RELAY,0\n")
            time.sleep(0.05)
            conn.close()
        except Exception:
            pass
    st.session_state.serial_conn = None


def parse_telemetry(line: str):
    parts = line.strip().split(",")
    if len(parts) != 8 or parts[0] != "DATA":
        return None
    try:
        return {
            "ms": int(parts[1]),
            "vib_g": float(parts[2]),
            "chip_temp_c": float(parts[3]),
            "rpm": float(parts[4]),
            "relay": int(parts[5]),
            "pulses": int(parts[6]),
            "sensor_ok": int(parts[7]),
            "received_at": time.time(),
        }
    except (ValueError, IndexError):
        return None


def connect(port: str):
    close_connection()
    conn = serial.Serial(port, BAUD_RATE, timeout=0.03, write_timeout=0.2)
    time.sleep(2.0)  # Uno commonly resets when its serial port opens.
    conn.reset_input_buffer()
    st.session_state.serial_conn = conn
    st.session_state.telemetry = []
    st.session_state.baseline = None
    st.session_state.high_count = 0
    st.session_state.critical_count = 0
    st.session_state.protection_latched = False
    st.session_state.last_error = ""
    send_command("PING")


st.title("⚙️ Motor Guardian")
st.caption("Live Arduino telemetry · MPU6050 vibration · IR tachometer · relay protection")

with st.sidebar:
    st.header("Hardware connection")
    available_ports = [p.device for p in list_ports.comports()]
    default_port = available_ports[0] if available_ports else "COM3"
    port = st.text_input("Arduino serial port", value=default_port,
                         help="Windows example: COM3. Linux example: /dev/ttyACM0.")
    c1, c2 = st.columns(2)
    if c1.button("Connect", use_container_width=True):
        try:
            connect(port.strip())
            st.success(f"Connected to {port}")
        except Exception as exc:
            st.session_state.last_error = str(exc)
            st.error(f"Connection failed: {exc}")
    if c2.button("Disconnect", use_container_width=True):
        close_connection()
        st.rerun()

    st.divider()
    st.subheader("Protection")
    st.session_state.auto_protection = st.toggle(
        "Enable automatic anomaly stop",
        value=st.session_state.auto_protection,
        help="Demo protection: stops the low-voltage fan after 3 consecutive vibration readings above 6 baseline standard deviations."
    )
    if st.button("Reset protection latch", use_container_width=True):
        st.session_state.protection_latched = False
        st.session_state.high_count = 0
        st.session_state.critical_count = 0

    st.caption("The Arduino turns the relay OFF if USB heartbeat is lost for 3 seconds or the MPU6050 is unavailable.")

if st.session_state.last_error:
    st.warning(f"Serial message: {st.session_state.last_error}")

conn = st.session_state.serial_conn
connected = conn is not None and conn.is_open

if not connected:
    st.info("Connect the Arduino to USB, select its serial port, then press Connect.")
    st.markdown("""
    **Before connecting**
    1. Upload `motor_guardian.ino` to the Arduino Uno.
    2. Close Arduino IDE Serial Monitor / Serial Plotter (only one program can use the port).
    3. Run this dashboard and select the same COM port.
    """)
    st.stop()


@st.fragment(run_every="500ms")
def live_panel():
    conn = st.session_state.serial_conn
    if conn is None or not conn.is_open:
        st.error("Arduino disconnected. The Arduino firmware should turn the relay OFF after heartbeat timeout.")
        return

    # Heartbeat is required to keep the Arduino relay-control session alive.
    try:
        conn.write(b"PING\n")
        while conn.in_waiting:
            line = conn.readline().decode("utf-8", errors="ignore").strip()
            record = parse_telemetry(line)
            if record is not None:
                st.session_state.telemetry.append(record)
                st.session_state.telemetry = st.session_state.telemetry[-MAX_ROWS:]
    except (serial.SerialException, OSError) as exc:
        st.session_state.last_error = str(exc)
        st.error(f"Serial communication lost: {exc}")
        return

    data = st.session_state.telemetry
    if not data:
        st.info("Waiting for Arduino telemetry...")
        return

    df = pd.DataFrame(data)
    latest = df.iloc[-1]
    sensor_ok = int(latest["sensor_ok"]) == 1

    # Learn baseline from the first 60 valid measurements.
    if st.session_state.baseline is None and len(df) >= BASELINE_SAMPLES:
        first = df.head(BASELINE_SAMPLES)
        if first["sensor_ok"].eq(1).all():
            mu = float(first["vib_g"].mean())
            sd = float(first["vib_g"].std(ddof=1))
            st.session_state.baseline = (mu, max(sd, 0.0005))

    baseline = st.session_state.baseline
    z_score = None
    status = "CALIBRATING" if baseline is None else "NORMAL"
    if baseline is not None:
        mu, sd = baseline
        z_score = (float(latest["vib_g"]) - mu) / sd
        if z_score >= WARNING_Z:
            status = "VIBRATION ANOMALY"
            st.session_state.high_count += 1
        else:
            st.session_state.high_count = 0

        if z_score >= CRITICAL_Z:
            st.session_state.critical_count += 1
        else:
            st.session_state.critical_count = 0

        if (st.session_state.auto_protection and
                not st.session_state.protection_latched and
                st.session_state.critical_count >= PERSIST):
            send_command("RELAY,0")
            st.session_state.protection_latched = True

    # Physical sensor failure is not a valid operating state.
    if not sensor_ok:
        status = "SENSOR FAULT"
        send_command("RELAY,0")

    if st.session_state.protection_latched:
        status = "PROTECTION STOPPED"

    # Operator controls. Relay commands are handled by the Arduino.
    st.subheader("Operator controls")
    b1, b2, b3 = st.columns(3)
    if b1.button("▶ Start fan", use_container_width=True,
                  disabled=(not sensor_ok or st.session_state.protection_latched)):
        if send_command("RELAY,1"):
            st.toast("Start command sent to Arduino.")
    if b2.button("■ Stop fan", use_container_width=True):
        send_command("RELAY,0")
        st.toast("Stop command sent to Arduino.")
    if b3.button("Clear event history", use_container_width=True):
        st.session_state.telemetry = []
        st.session_state.baseline = None
        st.session_state.high_count = 0
        st.session_state.critical_count = 0
        st.session_state.protection_latched = False
        st.rerun()

    st.divider()
    st.subheader("Live status")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Motor relay", "ON" if int(latest["relay"]) else "OFF")
    m2.metric("Vibration", f'{latest["vib_g"]:.5f} g')
    m3.metric("Fan speed", f'{latest["rpm"]:.0f} RPM')
    m4.metric("MPU6050 chip temperature", f'{latest["chip_temp_c"]:.1f} °C')

    if status == "NORMAL":
        st.success("NORMAL — no vibration anomaly detected.")
    elif status == "CALIBRATING":
        st.info(f"Learning healthy baseline: {min(len(df), BASELINE_SAMPLES)}/{BASELINE_SAMPLES} samples. Keep the fan in a known healthy condition.")
    elif status == "VIBRATION ANOMALY":
        st.warning(f"Vibration deviation: {z_score:.1f} standard deviations from baseline.")
    elif status == "SENSOR FAULT":
        st.error("MPU6050 unavailable. The Arduino has been commanded to switch the relay OFF.")
    else:
        st.error("PROTECTION STOP — vibration anomaly persisted. Inspect the fan before restarting.")

    if baseline is not None:
        st.caption(f"Vibration baseline: mean={baseline[0]:.5f} g, standard deviation={baseline[1]:.5f} g. "
                   "This is an acceleration-based demo metric, not mm/s vibration velocity.")

    chart = df.tail(240).set_index("received_at")[["vib_g", "rpm"]]
    st.subheader("Live sensor history")
    st.line_chart(chart, height=260)

    st.subheader("Latest telemetry")
    display = df.tail(10)[["ms", "vib_g", "chip_temp_c", "rpm", "relay", "pulses", "sensor_ok"]].iloc[::-1]
    st.dataframe(display, hide_index=True, use_container_width=True)

    st.caption("Current, phase imbalance, acoustic noise and insulation resistance are not measured by this hardware configuration; they are intentionally not shown as live readings.")


live_panel()
