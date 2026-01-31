# src/models/train_lineup_stability_model.py
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = PROJECT_ROOT / "data/features/feature_mart.csv"
MODEL_DIR = PROJECT_ROOT / "data/models"
OUT_DIR = PROJECT_ROOT / "data/outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["game_id", "playerNameI", "period"]

def main():
    df = pd.read_csv(FEATURE_PATH)

    # pick target = points-like proxy
    points_col = None
    for c in df.columns:
        if c.lower() in ["points", "period_points", "possession_points"]:
            points_col = c
            break
    if points_col is None:
        proxy_cols = [c for c in df.columns if "shots" in c.lower() or "makes" in c.lower() or "score" in c.lower()]
        df["_points_proxy"] = df[proxy_cols].sum(axis=1) if proxy_cols else 0
        points_col = "_points_proxy"

    # emphasize lineup features (still train with all numeric)
    drop_cols = set(KEYS + [points_col])
    X = df[[c for c in df.columns if c not in drop_cols]].select_dtypes(include=["number"]).fillna(0)
    y = df[points_col].fillna(0)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=df["game_id"]))

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = model.predict(X.iloc[test_idx])

    mae = mean_absolute_error(y.iloc[test_idx], preds)
    joblib.dump(model, MODEL_DIR / "lineup_impact_model.pkl")

    out = df.iloc[test_idx][KEYS].copy()
    out["y_true_points"] = y.iloc[test_idx].values
    out["y_pred_points"] = preds
    out.to_csv(OUT_DIR / "lineup_impact_predictions.csv", index=False)

    print("✅ Day 17 complete")
    print(f"MAE={mae:.4f}")

if __name__ == "__main__":
    main()
