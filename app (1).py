"""Motor Guardian dashboard (uses core.py as its engine).

Run:   streamlit run app.py
Needs: core.py in the SAME folder as this file.

Data sources (choose in the sidebar):
  1. Simulation          - the digital twin from core.py (all channels simulated)
  2. Arduino test        - FAKE 'DATA,...' lines pushed through the real parser (no hardware needed)
  3. Arduino USB         - real lines from the serial port (needs pyserial + a connected board)
"""
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

import core

BASELINE_N = 60
SRC_SIM = "Simulation (digital twin)"
SRC_FAKE = "Arduino test (fake lines, no hardware)"
SRC_USB = "Arduino USB (real hardware)"
SOURCES = [SRC_SIM, SRC_FAKE, SRC_USB]
ARDUINO_CHANNELS = {"vib", "temp"}          # diagnosis channels the Arduino line provides
ARDUINO_EXTRA = {"amb", "hum"}
FAKE_FAULTS = ["Healthy", "Bearing wear (vibration + heat rise)", "Overheating (heat only)"]
# Neutral values for channels the Arduino does NOT measure. They are never shown as readings;
# they only keep core.py's maths from failing on empty columns.
PLACEHOLDER = dict(curr=1.0, noise=1.0, rpm=1.0, res=1.0, unb=0.0)
LEVEL = {0: "HEALTHY", 1: "WARNING", 2: "CRITICAL"}
ICON = {"NORMAL": "🟢", "HIGH": "🔺", "LOW": "🔻", "UNBALANCED": "⚠️"}

st.set_page_config(page_title="Motor Guardian", page_icon="⚙️", layout="wide")
ss = st.session_state


# ---------------------------------------------------------------- state
def init(source):
    ss.source = source
    ss.df = pd.DataFrame({c: pd.Series(dtype=float) for c in core.COLS})
    ss.t, ss.t0, ss.fault = 0, 0, "Healthy"
    ss.base, ss.diag, ss.states, ss.z, ss.health = None, None, {}, None, None
    ss.diagnoser = core.Diagnoser()
    ss.events, ss.key, ss.hist = [], (None, 0), []
    ss.bad_lines, ss.good_lines, ss.start = 0, 0, time.time()
    ss.run_flag = None
    if "ser" not in ss:
        ss.ser, ss.serial_msg = None, "Not connected"


if "df" not in ss:
    init(SRC_SIM)


def channel_status(v):
    """Where does channel v come from right now?"""
    if ss.source == SRC_SIM:
        return "SIMULATED"
    if v in ARDUINO_CHANNELS or v in ARDUINO_EXTRA:
        return "REAL" if ss.source == SRC_USB else "TEST DATA (fake)"
    return "not connected"


def connected(v):
    return channel_status(v) != "not connected"


def unit(v):
    if v == "vib" and ss.source != SRC_SIM:
        return "g"
    return core.INFO[v][1]


def label(v):
    return core.INFO[v][0]


# ---------------------------------------------------------------- data in
def push(row, t):
    """Add one reading, update baseline + diagnosis."""
    ss.df.loc[len(ss.df)] = [t] + [row.get(c, np.nan) for c in core.COLS[1:]]
    if len(ss.df) > 1200:
        ss.df = ss.df.tail(1000).reset_index(drop=True)
    if ss.base is None and len(ss.df) >= BASELINE_N:
        ss.base = core.learn_baseline(ss.df.head(BASELINE_N))
    if ss.base is None:
        return
    raw, ss.states, ss.z, ss.health = core.evaluate(ss.df, ss.base)
    ss.diag = ss.diagnoser.update(raw)
    d = ss.diag
    key = (d["code"], d["level"]) if d else (None, 0)
    if key != ss.key:
        ss.events.append(dict(time=round(t, 1), code=d["code"] if d else "-",
                              event=d["name"] if d else "Back to normal",
                              level=LEVEL[d["level"]] if d else "RECOVERED",
                              action=d["action"] if d else "-", health=ss.health))
        ss.key = key
    ss.hist.append(dict(t=t, health=ss.health))
    ss.hist = ss.hist[-1000:]


def step_sim():
    k = ss.t - ss.t0 if ss.fault != "Healthy" else 0
    push(core.sample(k, ss.fault, ss.t), ss.t)
    ss.t += 1


def make_fake_line():
    """Build a text line exactly like the Arduino would send. Every 25th line is garbage on purpose."""
    k = max(0, ss.t - ss.t0) if ss.fault != "Healthy" else 0
    vib = 0.05 + core.rng.normal(0, 0.004)
    temp = 40 + core.rng.normal(0, 0.3)
    if ss.fault.startswith("Bearing"):
        vib += 0.002 * k
        temp += 0.03 * k
    elif ss.fault.startswith("Overheating"):
        temp += 0.15 * k
    if ss.t % 25 == 24:
        return "garbage#line"
    return f"DATA,{int(ss.t * 1000)},{vib:.4f},{temp:.2f},{24 + core.rng.normal(0, 0.1):.1f},{45 + core.rng.normal(0, 0.5):.1f},1"


def handle_line(line):
    p = core.parse_line(line)
    if p is None:
        if line.strip():
            ss.bad_lines += 1
        return
    ss.good_lines += 1
    try:
        t = int(line.strip().split(",")[1]) / 1000.0
    except (ValueError, IndexError):
        t = time.time() - ss.start
    ss.run_flag = p.pop("run", None)
    row = dict(PLACEHOLDER)
    row.update(p)
    push(row, t)


def step_fake():
    handle_line(make_fake_line())
    ss.t += 1


def step_usb():
    if ss.ser is None:
        return
    try:
        for _ in range(50):
            if not ss.ser.in_waiting:
                break
            handle_line(ss.ser.readline().decode(errors="ignore"))
    except Exception as e:
        ss.serial_msg, ss.ser = f"Serial error: {e}", None


def sample_interval_s():
    """Seconds between readings (None for simulation, where 1 tick = 1 simulated minute)."""
    if ss.source == SRC_SIM or len(ss.df) < 5:
        return None
    d = np.diff(ss.df["t"].tail(20).to_numpy())
    d = d[d > 0]
    return float(np.median(d)) if len(d) else None


def fmt_eta(eta):
    """core.eta_to_limit returns ticks. Convert using the real sampling interval."""
    if eta is None:
        return None
    iv = sample_interval_s()
    if ss.source == SRC_SIM:
        return f"~{eta:.0f} simulated minutes"
    if iv is None:
        return None
    sec = eta * iv
    return f"~{sec:.0f} seconds" if sec < 120 else f"~{sec / 60:.0f} minutes"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Control panel")
    src = st.radio("Data source", SOURCES, index=SOURCES.index(ss.source))
    if src != ss.source:
        init(src)

    if ss.source == SRC_USB:
        st.caption("Needs `pip install pyserial` and the Arduino plugged in.")
        port = st.text_input("Serial port", "COM3", help="Windows: COM3, COM4... Mac/Linux: /dev/ttyUSB0 or /dev/ttyACM0")
        baud = st.number_input("Baud rate", value=9600, step=1200,
                               help="Must equal Serial.begin(...) in the Arduino sketch.")
        c1, c2 = st.columns(2)
        if c1.button("Connect"):
            try:
                import serial
                ss.ser = serial.Serial(port, int(baud), timeout=0.05)
                ss.serial_msg = f"Connected to {port} @ {int(baud)}"
            except ImportError:
                ss.serial_msg = "pyserial is not installed (pip install pyserial)"
            except Exception as e:
                ss.serial_msg = f"Could not open {port}: {e}"
        if c2.button("Disconnect") and ss.ser is not None:
            ss.ser.close()
            ss.ser, ss.serial_msg = None, "Disconnected"
        st.write(ss.serial_msg)
    else:
        options = core.FAULTS if ss.source == SRC_SIM else FAKE_FAULTS
        idx = options.index(ss.fault) if ss.fault in options else 0
        locked = ss.base is None
        choice = st.selectbox("Inject fault", options, index=idx, disabled=locked,
                              help="Locked until the healthy baseline is learned." if locked else None)
        if choice != ss.fault:
            ss.fault, ss.t0 = choice, ss.t

    run = st.checkbox("Run", value=True)
    delay = st.slider("Seconds between updates", 0.05, 2.0, 0.25 if ss.source != SRC_USB else 0.5, 0.05)
    if st.button("Reset everything"):
        init(ss.source)
        st.rerun()

# ---------------------------------------------------------------- advance one step
if run:
    {SRC_SIM: step_sim, SRC_FAKE: step_fake, SRC_USB: step_usb}[ss.source]()

# ---------------------------------------------------------------- header + data-source badge
st.title("⚙️ Motor Guardian")
if ss.source == SRC_SIM:
    st.warning("🧪 **SIMULATED DATA** — every value comes from the digital twin, not from a real motor.")
elif ss.source == SRC_FAKE:
    st.warning("🧪 **TEST DATA** — fake Arduino lines used to test the dashboard. Not real measurements.")
else:
    if ss.ser is None:
        st.error("🔌 **Arduino USB selected, but not connected.** Use Connect in the sidebar.")
    else:
        st.success("📡 **REAL HARDWARE DATA** — only vibration, temperature, ambient temperature and humidity are measured.")
if ss.source != SRC_SIM:
    st.info("Limited diagnosis: current, noise, speed and insulation are **not measured**, so fault codes assume "
            "those are normal. Treat codes as indications, not confirmed diagnoses.")

tabs = st.tabs(["Overview", "Live Monitoring", "Fault Detection", "Action & Maintenance", "Analytics", "System Info"])
n = len(ss.df)
calibrating = ss.base is None
diag = ss.diag
last = ss.df.iloc[-1] if n else None
eta = None if (calibrating or diag is None) else core.eta_to_limit(ss.df, ss.base)
eta_txt = fmt_eta(eta)

# ---- A. Overview
with tabs[0]:
    if n == 0:
        st.info("Waiting for data…" if ss.source == SRC_USB else "Press Run to start.")
    elif calibrating:
        st.info(f"Learning the healthy baseline… {n}/{BASELINE_N} readings. "
                "Keep the motor in a known-healthy state until this finishes.")
        st.progress(min(n / BASELINE_N, 1.0))
    else:
        if diag is None:
            st.success("🟢 **HEALTHY** — no active fault codes.")
        elif diag["level"] == 1:
            st.warning(f"🟡 **WARNING · {diag['code']} · {diag['name']}**  \n**Action:** {diag['action']}")
        else:
            st.error(f"🔴 **CRITICAL · {diag['code']} · {diag['name']}**  \n**Action:** {diag['action']}")
        c = st.columns(4)
        c[0].metric("Condition", LEVEL[diag["level"]] if diag else "HEALTHY")
        c[1].metric("Health score", f"{ss.health}/100")
        c[2].metric("Active code", diag["code"] if diag else "none")
        verdict = core.decide(diag, eta, 60)
        c[3].metric("Suggested decision", verdict[0])
        st.caption(verdict[1] + " (Advice only. The dashboard does not control the motor.)")
    if n:
        st.subheader("Measurements")
        cols = st.columns(3)
        for i, v in enumerate(core.VARS):
            with cols[i % 3]:
                if connected(v):
                    st.metric(f"{label(v)} ({unit(v)})", f"{last[v]:.2f}")
                    st.caption(channel_status(v))
                else:
                    st.metric(label(v), "—")
                    st.caption("not connected")

# ---- B. Live monitoring
with tabs[1]:
    shown = [v for v in core.VARS if connected(v)]
    if n < 2:
        st.info("No data yet.")
    else:
        pick = st.selectbox("Inspect one measurement", ["All"] + shown, format_func=lambda v: v if v == "All" else label(v))
        view = ss.df.tail(240).set_index("t")
        for v in (shown if pick == "All" else [pick]):
            d = view[[v]].copy()
            d.columns = [f"{label(v)} ({unit(v)})"]
            if not calibrating:
                d["healthy baseline"] = ss.base["mu"][v]
                for side in ("hi", "lo"):
                    if v in ss.base["limits"][side]:
                        d["engineering limit"] = ss.base["limits"][side][v]
            st.markdown(f"**{label(v)}** · {channel_status(v)}")
            st.line_chart(d, height=200 if pick != "All" else 150)
        if ss.source != SRC_SIM and {"amb", "hum"} <= set(view.columns) and view["amb"].notna().any():
            st.markdown(f"**Ambient** · {channel_status('amb')}")
            st.line_chart(view[["amb", "hum"]], height=150)
        if ss.source != SRC_SIM:
            st.caption("Engineering limits are derived from the learned baseline. Replace them with datasheet values.")

# ---- C. Fault detection
with tabs[2]:
    if calibrating:
        st.info("Fault detection starts after the baseline is learned.")
    else:
        if diag is None:
            st.success("No active fault.")
        else:
            st.subheader(f"{diag['code']} · {diag['name']} — {LEVEL[diag['level']]}")
            st.markdown(f"**What it means:** {diag['why']}")
            st.markdown("**Affected measurements:** " + ", ".join(
                label(v) if v in core.INFO else "Phase imbalance" for v in diag["bad"]))
        rows = []
        for v in core.VARS:
            if connected(v):
                rows.append({"Measurement": label(v), "Value": f"{last[v]:.2f} {unit(v)}",
                             "Baseline": f"{ss.base['mu'][v]:.2f}", "Deviation (σ)": round(float(ss.z[v]), 1),
                             "State": f"{ICON[ss.states[v]]} {ss.states[v]}"})
            else:
                rows.append({"Measurement": label(v), "Value": "not connected", "Baseline": "—",
                             "Deviation (σ)": None, "State": "—"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(f"A measurement is HIGH/LOW when it stays ≥ {core.Z_STATE}σ from baseline for {core.PERSIST} readings in a row.")
        with st.expander("Fault code table"):
            st.dataframe(pd.DataFrame([{"Code": r["code"], "Name": r["name"], "Needs": str(r["need"]),
                                        "Meaning": r["why"]} for r in core.RULES + [core.UNKNOWN]]),
                         hide_index=True)

# ---- D. Action & maintenance (basic version; full tracking = Milestone 3)
with tabs[3]:
    if calibrating or diag is None:
        st.success("No action needed right now.")
    else:
        r = core.RULE_BY_CODE[diag["code"]]
        st.subheader(f"{diag['code']} · {diag['name']} — {LEVEL[diag['level']]}")
        st.markdown(f"**Problem:** {diag['why']}")
        st.markdown(f"**Recommended action:** {diag['action']}")
        st.markdown(f"**Recommended response time:** {diag['urgency']}")
        if eta_txt:
            st.markdown(f"**Estimated time to engineering limit:** {eta_txt}")
            st.caption("Assumption: straight-line trend of the last 40 readings continues unchanged. "
                       "Rough estimate only; accelerating faults will reach the limit sooner.")
        else:
            st.markdown("**Estimated time to engineering limit:** no reliable estimate "
                        "(not enough history, no clear trend, or no sampling interval known).")
    st.info("Coming in the next milestone: Pending / In Progress / Resolved status, technician notes, "
            "maintenance history, and a more careful time-to-limit estimate.")

# ---- E. Analytics
with tabs[4]:
    if not ss.hist:
        st.info("Analytics appear after the baseline is learned.")
    else:
        st.markdown("**Health score over time**")
        st.line_chart(pd.DataFrame(ss.hist).set_index("t"), height=200)
        st.markdown("**Event / fault history** (this session only; saved history comes later)")
        if ss.events:
            st.dataframe(pd.DataFrame(ss.events[::-1]), hide_index=True)
        else:
            st.caption("No events yet.")
        if not calibrating:
            st.markdown("**Statistics (last 240 readings)**")
            cols = [v for v in core.VARS if connected(v)]
            st.dataframe(ss.df[cols].tail(240).describe().T[["mean", "min", "max", "std"]].round(3),
                         )

# ---- F. System info
with tabs[5]:
    st.markdown(f"**Data source:** {ss.source}")
    if ss.source == SRC_USB:
        st.markdown(f"**Arduino connection:** {'🟢 ' if ss.ser else '🔴 '}{ss.serial_msg}")
        st.markdown(f"**Lines parsed OK:** {ss.good_lines} · **Lines ignored (malformed/other):** {ss.bad_lines}")
        st.caption("Expected line format (provisional until the sketch exists): DATA,ms,vib_g,temp_c,amb_c,hum_pct,run")
    elif ss.source == SRC_FAKE:
        st.markdown("**Arduino connection:** none (fake lines generated inside the app)")
        st.markdown(f"**Lines parsed OK:** {ss.good_lines} · **Garbage lines correctly ignored:** {ss.bad_lines}")
    else:
        st.markdown("**Arduino connection:** not used")
    st.dataframe(pd.DataFrame([{"Channel": label(v), "Status": channel_status(v)} for v in core.VARS]),
                 hide_index=True)
    st.markdown(f"**Baseline:** {'learned' if not calibrating else f'learning {n}/{BASELINE_N}'}")
    st.markdown(f"**Fault detection:** {'active' if not calibrating else 'waiting for baseline'}")
    iv = sample_interval_s()
    st.markdown("**Sampling interval:** " + ("simulated: 1 reading = 1 minute" if ss.source == SRC_SIM
                                            else f"{iv:.2f} s" if iv else "unknown yet"))
    if iv and iv > 3.3:
        st.warning("Faults need 3 consecutive readings to be flagged. At this interval detection takes "
                   f"~{3 * iv:.0f} s, which is longer than the 10 s target.")
    st.markdown(f"**Settings:** z-threshold {core.Z_STATE}σ · persistence {core.PERSIST} readings · "
                f"alarm hold {core.HOLD} · clear after {core.CLEAR_TICKS} normal readings")
    st.markdown("**Motor START/STOP control:** not implemented (waiting for confirmed hardware and safety design). "
                "This dashboard is not a safety-certified emergency stop.")

# ---------------------------------------------------------------- loop
if run and not os.environ.get("MG_NO_LOOP"):
    time.sleep(delay)
    st.rerun()
