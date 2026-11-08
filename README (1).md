# Motor Guardian

Self-diagnosing factory-motor monitor (IoT & AI hackathon). Watches temperature, current, vibration, noise, RPM and insulation resistance, classifies each as LOW / NORMAL / HIGH (plus UNBALANCED for phase currents), and maps the pattern to a printer-style fault code with a recommended action.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## How it works
1. Learn a healthy baseline from the first 60 samples.
2. Per variable: z-score vs baseline, state = HIGH/LOW if |z| >= 3 for 3 consecutive samples.
3. State pattern -> fault code (E01 insulation, E02 phase imbalance, E03 overload/jam, E04 bearing wear, E05 cooling failure, E06 vibration anomaly, E07 under-load, E99 unclassified).
4. WARNING on state change; CRITICAL when an engineering limit is reached -> protection action.

## Notes
- Data is simulated (digital twin). Thresholds are demo values; replace with motor datasheet limits.
- Hardware version (Arduino + MPU6050 + IR + relay) is run locally, not on the cloud deployment.
