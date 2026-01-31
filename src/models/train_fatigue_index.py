# src/models/train_fatigue_index.py
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = PROJECT_ROOT / "data/features/feature_mart.csv"
MODEL_DIR = PROJECT_ROOT / "data/models"
OUT_DIR = PROJECT_ROOT / "data/outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["game_id", "playerNameI", "period"]

def main():
    df = pd.read_csv(FEATURE_PATH)
    for k in KEYS:
        if k not in df.columns:
            raise ValueError(f"feature_mart.csv missing key: {k}")

    # Choose a "points this period" column if it exists; otherwise create a proxy.
    # If you have a possession points column merged in, map it here.
    points_col = None
    for c in df.columns:
        if c.lower() in ["points", "period_points", "possession_points"]:
            points_col = c
            break

    if points_col is None:
        # fallback: use usage shots/makes as a proxy
        proxy_cols = [c for c in df.columns if "shots" in c.lower() or "makes" in c.lower() or "score" in c.lower()]
        if not proxy_cols:
            raise ValueError("No points-like or proxy columns found to build target.")
        df["_period_points_proxy"] = df[proxy_cols].sum(axis=1)
        points_col = "_period_points_proxy"

    # Build next-period delta target per (game, player)
    df = df.sort_values(["game_id", "playerNameI", "period"])
    df["next_points"] = df.groupby(["game_id", "playerNameI"])[points_col].shift(-1)
    df["fatigue_target_drop"] = df[points_col] - df["next_points"]  # positive => drop next period

    train_df = df.dropna(subset=["fatigue_target_drop"]).copy()

    # Features: drop keys + target + helper cols
    drop_cols = set(KEYS + ["next_points", "fatigue_target_drop"])
    X = train_df[[c for c in train_df.columns if c not in drop_cols]]
    y = train_df["fatigue_target_drop"]

    # Keep only numeric
    X = X.select_dtypes(include=["number"]).fillna(0)

    # Leakage-safe split by game_id
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    (train_idx, test_idx) = next(splitter.split(X, y, groups=train_df["game_id"]))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    joblib.dump(model, MODEL_DIR / "fatigue_model.pkl")

    out = train_df.iloc[test_idx][KEYS].copy()
    out["y_true_drop"] = y_test.values
    out["y_pred_drop"] = preds
    out.to_csv(OUT_DIR / "fatigue_predictions.csv", index=False)

    metrics = pd.DataFrame([{"model": "fatigue", "MAE": mae, "RMSE": rmse}])
    metrics.to_csv(PROJECT_ROOT / "data/reports/fatigue_metrics.csv", index=False)

    print("✅ Day 15 complete")
    print(f"MAE={mae:.4f} RMSE={rmse:.4f}")

if __name__ == "__main__":
    main()
