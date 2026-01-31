# src/decision_support/narratives.py
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    fatigue = pd.read_csv(PROJECT_ROOT / "data/outputs/fatigue_predictions.csv")
    clutch = pd.read_csv(PROJECT_ROOT / "data/outputs/clutch_predictions.csv") if (PROJECT_ROOT / "data/outputs/clutch_predictions.csv").exists() else None

    insights = []

    # Top fatigue risk: biggest predicted drop
    top_fatigue = fatigue.sort_values("y_pred_drop", ascending=False).head(10)
    for _, r in top_fatigue.iterrows():
        insights.append({
            "type": "fatigue_risk",
            "game_id": r["game_id"],
            "player": r["playerNameI"],
            "period": r["period"],
            "insight": f"High fatigue risk: model predicts performance drop of {r['y_pred_drop']:.2f} next period."
        })

    if clutch is not None and not clutch.empty:
        top_clutch = clutch.sort_values("y_proba", ascending=False).head(10)
        for _, r in top_clutch.iterrows():
            insights.append({
                "type": "clutch_opportunity",
                "game_id": r["game_id"],
                "player": "",
                "period": 4,
                "insight": f"High clutch scoring probability at clock {r['clock_sec']:.0f}s (p={r['y_proba']:.2f})."
            })

    out = pd.DataFrame(insights)
    out.to_csv(OUT_DIR / "coach_insights.csv", index=False)
    print("✅ Day 22 complete: data/outputs/coach_insights.csv")

if __name__ == "__main__":
    main()
