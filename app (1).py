"""Motor Guardian v2 - state-based self-diagnosis for a factory motor (printer-style fault codes).
Run:  pip install streamlit pandas numpy  &&  streamlit run motor_guardian_v2.py
Pipeline: sensors -> deviation from healthy baseline (z-score) -> state per variable
          (LOW / NORMAL / HIGH, + UNBALANCED) -> rule table -> fault code + cause + action.
"""
import time
import numpy as np
import pandas as pd

# ---------------- variables ----------------
VARS = ["temp", "curr", "vib", "noise", "rpm", "res"]
INFO = {  # label, unit, signal group
    "temp": ("Temperature", "°C", "Thermal"),
    "curr": ("Current", "A", "Electrical"),
    "res": ("Insulation resistance", "MΩ", "Electrical"),
    "vib": ("Vibration", "mm/s", "Mechanical"),
    "rpm": ("Speed", "RPM", "Mechanical"),
    "noise": ("Noise", "dB", "Acoustic"),
}
BASELINE_N = 60            # ticks used to learn "healthy" (1 tick = 1 simulated minute)
Z_STATE = 3.0              # |z| >= 3 -> variable is HIGH or LOW
Z_EXIT = 1.5               # hysteresis: a flagged variable returns to NORMAL only below this (stops flicker)
PERSIST = 3                # state must hold this many consecutive ticks
UNB_WARN, UNB_EXIT, UNB_CRIT = 2.0, 1.5, 5.0            # phase-current imbalance, % (NEMA-style: >5% is not acceptable)
HARD_HI = {"temp": 90.0, "vib": 7.1, "curr": 24.0}   # engineering limits - replace with datasheet values
HARD_LO = {"res": 50.0, "rpm": 1420.0}
HEALTH_Z = 10.0
FAULTS = ["Healthy", "Bearing wear", "Overload / jam", "Cooling failure",
          "Phase imbalance", "Insulation degradation", "Rotor imbalance", "Sudden shock"]
rng = np.random.default_rng()

# ---------------- fault-code table (ordered by priority: first match wins) ----------------
RULES = [
    dict(code="E01", name="Insulation degradation", need={"res": "LOW"},
         warn="Schedule insulation test (megger) and inspect for moisture or contamination.",
         crit="TRIP: stop motor and lock out. Winding failure and fire risk."),
    dict(code="E02", name="Phase imbalance (supply or connection)", need={"unb": "UNBALANCED"},
         warn="Check supply voltage, terminals and connections on all three phases.",
         crit="TRIP: stop motor. Severe imbalance overheats windings."),
    dict(code="E03", name="Overload / mechanical jam", need={"curr": "HIGH", "temp": "HIGH"},
         warn="Check the driven machine (jam, belt, blocked pump) and reduce load.",
         crit="Reduce load automatically; trip if thermal limit is reached."),
    dict(code="E04", name="Bearing wear", need={"vib": "HIGH", "noise": "HIGH"},
         warn="Schedule bearing inspection and lubrication at the next planned stop.",
         crit="Derate speed 30% and plan immediate bearing replacement."),
    dict(code="E05", name="Cooling failure", need={"temp": "HIGH", "curr": "NORMAL"},
         warn="Check cooling fan, vents and ambient temperature.",
         crit="Reduce duty cycle and stop if temperature keeps rising."),
    dict(code="E06", name="Vibration anomaly (imbalance, misalignment, looseness, shock)",
         need={"vib": "HIGH", "noise": "NORMAL", "curr": "NORMAL"},
         warn="Check balance, alignment and mounting bolts at the next stop.",
         crit="Derate speed 30% and inspect immediately."),
    dict(code="E07", name="Under-load / decoupled load", need={"curr": "LOW", "rpm": "HIGH"},
         warn="Check coupling, belt or whether the driven machine is running empty.",
         crit="Stop motor and inspect the transmission."),
]
UNKNOWN = dict(code="E99", name="Unclassified anomaly", need={},
               warn="Pattern not in the fault table: inspect the flagged variables.",
               crit="Reduce load and inspect immediately.")


# ---------------- simulator (digital twin) ----------------
def sample(k, fault, t):
    """k = ticks since fault injection. Returns one row of sensor readings."""
    ia, ib, ic = (20 + rng.normal(0, 0.2) for _ in range(3))
    temp = 65 + 0.5 * np.sin(t / 60) + rng.normal(0, 0.6)
    vib = 2.0 + rng.normal(0, 0.12)
    noise = 70 + rng.normal(0, 0.8)
    rpm = 1480 + rng.normal(0, 5)
    res_decay = 1.0
    if fault.startswith("Bearing"):
        vib += 0.04 * k + 0.00008 * k ** 2
        noise += 0.15 * k
        temp += 0.04 * k
    elif fault.startswith("Overload"):
        up = 0.08 * min(k, 100)
        ia, ib, ic = ia + up, ib + up, ic + up
        temp += 0.15 * k
        rpm -= 0.6 * min(k, 80)
        vib += 0.01 * k
    elif fault.startswith("Cooling"):
        temp += 0.15 * k
    elif fault.startswith("Phase"):
        x = 0.03 * min(k, 100)
        ib, ia, ic = ib - x, ia + x / 2, ic + x / 2
        temp += 0.03 * k
    elif fault.startswith("Insulation"):
        res_decay = float(np.exp(-k / 60))
        temp += 0.02 * k
    elif fault.startswith("Rotor"):
        vib += 0.03 * min(k, 60)
    elif fault.startswith("Sudden") and k < 6:
        vib += 4.0 + rng.normal(0, 0.5)
    curr = (ia + ib + ic) / 3
    unb = max(abs(ia - curr), abs(ib - curr), abs(ic - curr)) / curr * 100
    res = 200 * res_decay + rng.normal(0, 4)
    return dict(temp=temp, curr=curr, vib=vib, noise=noise, rpm=rpm, res=res, unb=unb)


# ---------------- state estimation + diagnosis ----------------
def learn_baseline(df):
    b = df[VARS].head(BASELINE_N)
    mu = b.mean()
    sd = np.maximum(b.std(), 0.005 * mu.abs())   # floor avoids hypersensitivity
    return mu, sd


def get_states(df, base, prev=None):
    """Signed z-score vs healthy baseline -> LOW / NORMAL / HIGH. Enter after PERSIST ticks beyond
    Z_STATE; leave only after PERSIST ticks inside Z_EXIT (hysteresis)."""
    prev = prev or {}
    mu, sd = base
    z = (df[VARS].tail(PERSIST) - mu) / sd
    states = {}
    for v in VARS:
        if (z[v] >= Z_STATE).all():
            states[v] = "HIGH"
        elif (z[v] <= -Z_STATE).all():
            states[v] = "LOW"
        elif prev.get(v, "NORMAL") != "NORMAL" and not (z[v].abs() < Z_EXIT).all():
            states[v] = prev[v]
        else:
            states[v] = "NORMAL"
    u = df["unb"].tail(PERSIST)
    if (u >= UNB_WARN).all() or (prev.get("unb") == "UNBALANCED" and not (u < UNB_EXIT).all()):
        states["unb"] = "UNBALANCED"
    else:
        states["unb"] = "NORMAL"
    return states, z.iloc[-1]


def is_critical(vars_, z, last):
    for v in vars_:
        if v == "unb":
            if last["unb"] >= UNB_CRIT:
                return True
        elif (v in HARD_HI and last[v] >= HARD_HI[v]) or (v in HARD_LO and last[v] <= HARD_LO[v]):
            return True
    return False


def evaluate(df, base, prev_states=None, latched=False):
    """Returns (diagnosis or None, states, z_last, health 0-100).
    latched=True keeps CRITICAL until the incident clears (no flapping around a limit)."""
    states, z = get_states(df, base, prev_states)
    last = df.iloc[-1]
    bad = [v for v in VARS + ["unb"] if states[v] != "NORMAL"]
    zmax = max(abs(z[v]) for v in VARS)
    health = int(round(100 * (1 - min(max(zmax / HEALTH_Z, last["unb"] / UNB_CRIT), 1))))
    if not bad:
        return None, states, z, health
    rule = next((r for r in RULES if all(states[k] == s for k, s in r["need"].items())), UNKNOWN)
    vars_ = list(rule["need"]) or bad
    crit = latched or is_critical(bad if rule is UNKNOWN else vars_, z, last)
    diag = dict(code=rule["code"], name=rule["name"], level=2 if crit else 1, vars=bad,
                action=rule["crit"] if crit else rule["warn"])
    return diag, states, z, health


def eta_to_limit(df, n=40):
    """Linear-trend estimate of ticks until a hard limit (None if no clear trend)."""
    if len(df) < n:
        return None
    x, best = np.arange(n), None
    for col, lim in list(HARD_HI.items()) + [(c, l) for c, l in HARD_LO.items()]:
        y = df[col].tail(n).to_numpy()
        slope = np.polyfit(x, y, 1)[0]
        toward = slope > 0 if col in HARD_HI else slope < 0
        if toward and abs(slope) > 1e-3:
            eta = max(0.0, (lim - y[-1]) / slope)
            if eta < 600:
                best = eta if best is None else min(best, eta)
    return best


# ---------------- Streamlit UI ----------------
def main():
    import streamlit as st
    st.set_page_config(page_title="Motor Guardian v2", page_icon="⚙️", layout="wide")
    ss = st.session_state
    ICON = {"NORMAL": "🟢", "HIGH": "🔺", "LOW": "🔻", "UNBALANCED": "⚠️"}

    def init():
        ss.df = pd.DataFrame({c: pd.Series(dtype=float) for c in ["t"] + VARS + ["unb"]})
        ss.t, ss.fault, ss.t0 = 0, "Healthy", 0
        ss.base, ss.res, ss.key, ss.events, ss.states = None, None, (None, 0), [], {}

    if "df" not in ss:
        init()

    st.title("⚙️ Motor Guardian — self-diagnosing motor")
    st.caption("Simulated telemetry · state per variable · fault-code diagnosis · automatic protection")

    with st.sidebar:
        st.header("Control panel")
        choice = st.selectbox("Inject fault", FAULTS, index=FAULTS.index(ss.fault))
        if choice != ss.fault:
            ss.fault, ss.t0 = choice, ss.t
        run = st.checkbox("Run simulation", value=True)
        delay = st.slider("Seconds per tick", 0.05, 1.0, 0.2, 0.05)
        if st.button("Reset"):
            init()
            st.rerun()
        st.markdown(f"**State rule**  \n|z| ≥ {Z_STATE}σ for {PERSIST} ticks → HIGH/LOW  \n"
                    f"Imbalance ≥ {UNB_WARN}% → UNBALANCED  \nCritical = engineering limit hit")

    ph_banner, ph_kpi, ph_state, ph_chart, ph_log = (st.empty() for _ in range(5))

    def step():
        k = ss.t - ss.t0 if ss.fault != "Healthy" else 0
        row = sample(k, ss.fault, ss.t)
        ss.df.loc[len(ss.df)] = [ss.t] + [row[c] for c in VARS + ["unb"]]
        ss.t += 1
        if len(ss.df) > 1200:
            ss.df = ss.df.tail(1000).reset_index(drop=True)
        if ss.base is None and len(ss.df) >= BASELINE_N:
            ss.base = learn_baseline(ss.df)
        if ss.base is not None:
            ss.res = evaluate(ss.df, ss.base, ss.states, ss.key[1] == 2)
            ss.states = ss.res[1]
            diag = ss.res[0]
            key = (diag["code"], diag["level"]) if diag else (None, 0)
            if key != ss.key:
                if diag:
                    ss.events.append({"tick": ss.t, "code": diag["code"], "fault": diag["name"],
                                      "level": "WARNING" if diag["level"] == 1 else "CRITICAL",
                                      "action": diag["action"]})
                else:
                    ss.events.append({"tick": ss.t, "code": "-", "fault": "Back to normal",
                                      "level": "RECOVERED", "action": "-"})
                ss.key = key

    def render():
        df = ss.df
        if len(df) == 0:
            ph_banner.info("Press Run to start.")
            return
        if ss.base is None:
            ph_banner.info(f"Calibrating healthy baseline… {len(df)}/{BASELINE_N}")
            return
        diag, states, z, health = ss.res
        eta = eta_to_limit(df) if diag else None
        eta_txt = f"  \nTrend: engineering limit in ~{eta:.0f} min" if eta is not None else ""
        if diag is None:
            ph_banner.success("🟢 **READY** — no active fault codes.")
        elif diag["level"] == 1:
            ph_banner.warning(f"🟡 **{diag['code']} · {diag['name']}** — WARNING  \n"
                              f"**Action:** {diag['action']}{eta_txt}")
        else:
            ph_banner.error(f"🔴 **{diag['code']} · {diag['name']}** — CRITICAL  \n"
                            f"**Protection:** {diag['action']}{eta_txt}")
        last = df.iloc[-1]
        mu = ss.base[0]
        with ph_kpi.container():
            c = st.columns(3)
            c[0].metric("Health index", f"{health}/100")
            c[1].metric("Active code", diag["code"] if diag else "none")
            c[2].metric("Phase imbalance", f"{last['unb']:.1f}%")
        rows = []
        for v in VARS:
            lab, unit, grp = INFO[v]
            rows.append({"Signal": grp, "Variable": lab, "Value": f"{last[v]:.1f} {unit}",
                         "Healthy baseline": f"{mu[v]:.1f}", "Deviation (σ)": round(float(z[v]), 1),
                         "State": f"{ICON[states[v]]} {states[v]}"})
        rows.append({"Signal": "Electrical", "Variable": "Phase imbalance", "Value": f"{last['unb']:.1f} %",
                     "Healthy baseline": "< 2", "Deviation (σ)": None,
                     "State": f"{ICON[states['unb']]} {states['unb']}"})
        with ph_state.container():
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        view = df.tail(240).set_index("t")
        with ph_chart.container():
            for chunk in (VARS[:3], VARS[3:]):
                cols = st.columns(3)
                for col, v in zip(cols, chunk):
                    d = view[[v]].copy()
                    d["healthy"] = mu[v]
                    if v in HARD_HI:
                        d["limit"] = HARD_HI[v]
                    if v in HARD_LO:
                        d["limit"] = HARD_LO[v]
                    col.markdown(f"**{INFO[v][0]} ({INFO[v][1]})**")
                    col.line_chart(d, height=170)
        with ph_log.container():
            st.markdown("**Event log**")
            if ss.events:
                st.dataframe(pd.DataFrame(ss.events[::-1]).head(8), hide_index=True, use_container_width=True)
            else:
                st.caption("No events yet.")

    render()
    while run:
        step()
        render()
        time.sleep(delay)


if __name__ == "__main__":
    main()
