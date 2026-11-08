"""Motor Guardian core: simulator, state estimation, fault codes, decision logic, data helpers."""
import numpy as np
import pandas as pd

VARS = ["temp", "curr", "vib", "noise", "rpm", "res"]
EXTRA = ["unb", "amb", "hum"]
COLS = ["t"] + VARS + EXTRA
INFO = {  # label, unit (simulator units), signal group
    "temp": ("Temperature", "°C", "Thermal"),
    "curr": ("Current", "A", "Electrical"),
    "res": ("Insulation resistance", "MΩ", "Electrical"),
    "vib": ("Vibration", "mm/s", "Mechanical"),
    "rpm": ("Speed", "RPM", "Mechanical"),
    "noise": ("Noise", "dB", "Acoustic"),
}
Z_STATE = 3.0                    # |z| >= 3 -> HIGH / LOW
PERSIST = 3                      # state must hold this many consecutive ticks
UNB_WARN, UNB_CRIT = 2.0, 5.0    # phase-current imbalance in %
HEALTH_Z = 10.0
HOLD = 5                         # ticks a new fault code must persist before the diagnosis switches to it
CLEAR_TICKS = 10                 # ticks of "all normal" before an alert is cleared
SD_MIN = {"temp": 0.5}           # minimum sd per variable (avoids hypersensitivity on very clean signals)
FAULTS = ["Healthy", "Bearing wear", "Overload / jam", "Cooling failure",
          "Phase imbalance", "Insulation degradation", "Rotor imbalance", "Sudden shock"]
rng = np.random.default_rng()

# Fault-code table, ordered by priority (first match wins).
RULES = [
    dict(code="E01", name="Insulation degradation", need={"res": "LOW"},
         why="Moisture, heat or ageing is breaking down the winding insulation.",
         warn="Schedule an insulation test (megger) and inspect for moisture or contamination.",
         crit="TRIP: stop the motor and lock it out. Winding failure and fire risk.",
         urgency=("within 1 week", "immediately")),
    dict(code="E02", name="Phase imbalance (supply or connection)", need={"unb": "UNBALANCED"},
         why="Unequal phase currents: bad supply, loose terminal or a damaged phase.",
         warn="Check supply voltage, terminals and connections on all three phases.",
         crit="TRIP: stop the motor. Severe imbalance overheats the windings.",
         urgency=("within 24 h", "immediately")),
    dict(code="E03", name="Overload / mechanical jam", need={"curr": "HIGH", "temp": "HIGH"},
         why="The motor works harder than normal: jam, friction or an overloaded machine.",
         warn="Check the driven machine (jam, belt, blocked pump) and reduce load.",
         crit="Reduce load; stop if the thermal limit is reached.",
         urgency=("within 24 h", "now")),
    dict(code="E04", name="Bearing wear", need={"vib": "HIGH", "noise": "HIGH"},
         why="Worn or poorly lubricated bearing: vibration and noise rise together.",
         warn="Schedule bearing inspection and lubrication at the next planned stop.",
         crit="Derate and plan immediate bearing replacement.",
         urgency=("within 1-2 weeks", "within 24 h")),
    dict(code="E05", name="Cooling failure", need={"temp": "HIGH", "curr": "NORMAL"},
         why="Heat is not leaving the motor: blocked vents, failed fan or hot surroundings.",
         warn="Check the cooling fan, vents and ambient temperature.",
         crit="Reduce duty cycle and stop if temperature keeps rising.",
         urgency=("within 24 h", "now")),
    dict(code="E06", name="Vibration anomaly (imbalance, misalignment, looseness, shock)",
         need={"vib": "HIGH", "noise": "NORMAL", "curr": "NORMAL"},
         why="Mechanical imbalance, misalignment or loose mounting.",
         warn="Check balance, alignment and mounting bolts at the next stop.",
         crit="Derate and inspect immediately.",
         urgency=("next planned stop", "within 24 h")),
    dict(code="E07", name="Under-load / decoupled load", need={"curr": "LOW", "rpm": "HIGH"},
         why="The motor runs with little load: broken coupling or belt, or an empty machine.",
         warn="Check the coupling, belt, or whether the driven machine is running empty.",
         crit="Stop the motor and inspect the transmission.",
         urgency=("next shift", "now")),
]
UNKNOWN = dict(code="E99", name="Unclassified anomaly", need={},
               why="The pattern is not in the fault table.",
               warn="Inspect the flagged variables.",
               crit="Reduce load and inspect immediately.",
               urgency=("next shift", "now"))
RULE_BY_CODE = {r["code"]: r for r in RULES + [UNKNOWN]}
ELECTRICAL = {"E01", "E02"}


# ---------------- simulator (digital twin) ----------------
def sample(k, fault, t):
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
        vib += 3.5 + rng.normal(0, 0.5)
    curr = (ia + ib + ic) / 3
    unb = max(abs(ia - curr), abs(ib - curr), abs(ic - curr)) / curr * 100
    res = 200 * res_decay + rng.normal(0, 4)
    return dict(temp=temp, curr=curr, vib=vib, noise=noise, rpm=rpm, res=res, unb=unb,
                amb=np.nan, hum=np.nan)


# ---------------- real-data helpers ----------------
ALIASES = {"vib_g": "vib", "temp_c": "temp", "amb_c": "amb", "hum_pct": "hum"}


def parse_line(line):
    """Arduino line 'DATA,ms,vib_g,temp_c,amb_c,hum_pct,run' -> dict, or None if malformed."""
    p = line.strip().split(",")
    if len(p) != 7 or p[0] != "DATA":
        return None
    try:
        return dict(vib=float(p[2]), temp=float(p[3]), amb=float(p[4]), hum=float(p[5]), run=int(p[6]))
    except ValueError:
        return None


def normalize_recording(raw):
    """Recording CSV (from logger.py or your own) -> DataFrame with canonical column names."""
    df = raw.rename(columns=lambda c: ALIASES.get(str(c).strip().lower(), str(c).strip().lower()))
    keep = [c for c in VARS + EXTRA if c in df.columns]
    if "vib" not in keep or "temp" not in keep:
        raise ValueError("CSV needs at least vib (or vib_g) and temp (or temp_c) columns")
    out = df[keep].apply(pd.to_numeric, errors="coerce")
    return out.dropna(subset=["vib", "temp"]).reset_index(drop=True)


def merge_real(row, real):
    """Real readings overwrite the simulated ones; everything else stays simulated."""
    for k, v in real.items():
        if k in VARS + EXTRA and pd.notna(v):
            row[k] = float(v)
    return row


# ---------------- state estimation + diagnosis ----------------
def learn_baseline(b):
    mu = b[VARS].mean()
    sd = np.maximum(b[VARS].std(), 0.005 * mu.abs())
    for v, m in SD_MIN.items():
        sd[v] = max(sd[v], m)
    limits = {  # engineering limits derived from the learned healthy values (replace with datasheet values)
        "hi": {"temp": mu["temp"] + 25.0, "vib": mu["vib"] * 3.5, "curr": mu["curr"] * 1.2},
        "lo": {"res": mu["res"] * 0.25, "rpm": mu["rpm"] * 0.96},
    }
    return dict(mu=mu, sd=sd, limits=limits)


def get_states(df, base):
    z = (df[VARS].tail(PERSIST) - base["mu"]) / base["sd"]
    states = {}
    for v in VARS:
        states[v] = "HIGH" if (z[v] >= Z_STATE).all() else "LOW" if (z[v] <= -Z_STATE).all() else "NORMAL"
    states["unb"] = "UNBALANCED" if (df["unb"].tail(PERSIST) >= UNB_WARN).all() else "NORMAL"
    return states, z.iloc[-1]


def is_critical(vars_, last, limits):
    for v in vars_:
        if v == "unb":
            if last["unb"] >= UNB_CRIT:
                return True
        elif (v in limits["hi"] and last[v] >= limits["hi"][v]) or (v in limits["lo"] and last[v] <= limits["lo"][v]):
            return True
    return False


def evaluate(df, base):
    """Instantaneous diagnosis: (raw or None, states, z_last, health 0-100)."""
    states, z = get_states(df, base)
    last = df.iloc[-1]
    bad = [v for v in VARS + ["unb"] if states[v] != "NORMAL"]
    zmax = max(abs(z[v]) for v in VARS)
    health = int(round(100 * (1 - min(max(zmax / HEALTH_Z, last["unb"] / UNB_CRIT), 1))))
    if not bad:
        return None, states, z, health
    rule = next((r for r in RULES if all(states[k] == s for k, s in r["need"].items())), UNKNOWN)
    vars_ = list(rule["need"]) or bad
    crit = is_critical(vars_, last, base["limits"])
    return dict(code=rule["code"], level=2 if crit else 1, bad=bad), states, z, health


def make_diag(cur):
    r = RULE_BY_CODE[cur["code"]]
    lvl = cur["level"]
    return dict(code=r["code"], name=r["name"], level=lvl, bad=cur["bad"], why=r["why"],
                action=r["crit"] if lvl == 2 else r["warn"], urgency=r["urgency"][lvl - 1])


class Diagnoser:
    """Smooths the instantaneous diagnosis: level latches upward, codes switch only when stable."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur, self.cand, self.cand_n, self.clear_n = None, None, 0, 0

    def update(self, raw):
        if raw is None:
            if self.cur is not None:
                self.clear_n += 1
                if self.clear_n >= CLEAR_TICKS:
                    self.reset()
            return make_diag(self.cur) if self.cur else None
        self.clear_n = 0
        if self.cur is None:
            self.cur = dict(code=raw["code"], level=raw["level"], bad=raw["bad"])
        else:
            self.cur["level"] = max(self.cur["level"], raw["level"])
            self.cur["bad"] = raw["bad"]
            if raw["code"] != self.cur["code"]:
                self.cand_n = self.cand_n + 1 if raw["code"] == self.cand else 1
                self.cand = raw["code"]
                if self.cand_n >= HOLD:
                    self.cur["code"], self.cand, self.cand_n = raw["code"], None, 0
            else:
                self.cand, self.cand_n = None, 0
        return make_diag(self.cur)


def eta_to_limit(df, base, n=40):
    """Linear-trend estimate of ticks until an engineering limit (None if no clear trend)."""
    if len(df) < n:
        return None
    x, best = np.arange(n), None
    for side, sign in (("hi", 1), ("lo", -1)):
        for col, lim in base["limits"][side].items():
            y = df[col].tail(n).to_numpy()
            slope = np.polyfit(x, y, 1)[0] * sign
            if slope > 1e-6 * max(1.0, abs(lim)):
                eta = max(0.0, (lim - y[-1]) * sign / slope)
                if eta < 600:
                    best = eta if best is None else min(best, eta)
    return best


def decide(diag, eta, window):
    """Stop-or-continue rule. window = minutes until the next planned stop."""
    if diag is None:
        return "RUN", "No active fault."
    if diag["level"] == 2:
        if diag["code"] in ELECTRICAL:
            return "STOP NOW", "Electrical fault: fire and winding-damage risk."
        return "CONTROLLED STOP", "Engineering limit reached: stop at the next safe point and inspect."
    if eta is not None and eta < window:
        return "PLAN STOP", f"Trend reaches the limit in ~{eta:.0f} min, before the next planned stop ({window} min)."
    return "RUN + MONITOR", "Early deviation: schedule an inspection."
