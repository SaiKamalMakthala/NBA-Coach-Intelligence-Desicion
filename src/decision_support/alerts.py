# src/decision_support/alerts.py
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    fatigue = pd.read_csv(PROJECT_ROOT / "data/outputs/fatigue_predictions.csv")

    # Thresholds (tune later)
    thr = fatigue["y_pred_drop"].quantile(0.90)

    alerts = fatigue[fatigue["y_pred_drop"] >= thr].copy()
    alerts["severity"] = "HIGH"
    alerts["reason"] = "Predicted next-period drop in performance (fatigue risk)."

    alerts.to_csv(OUT_DIR / "alerts.csv", index=False)
    print("✅ Day 23 complete: data/outputs/alerts.csv")

if __name__ == "__main__":
    main()
