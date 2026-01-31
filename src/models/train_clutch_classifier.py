# src/models/train_clutch_classifier.py
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PBP_PATH = PROJECT_ROOT / "data/processed/pbp_with_possessions.csv"
MODEL_DIR = PROJECT_ROOT / "data/models"
OUT_DIR = PROJECT_ROOT / "data/outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    pbp = pd.read_csv(PBP_PATH)

    # clutch filter: period 4 and last 300 sec; score diff <= 5
    # scoreHome/scoreAway exist in your schema
    pbp["score_diff"] = (pbp["scoreHome"] - pbp["scoreAway"]).abs()
    clutch = pbp[(pbp["period"] == 4) & (pbp["clock_sec"] <= 300) & (pbp["score_diff"] <= 5)].copy()

    if clutch.empty:
        raise ValueError("No clutch rows found with the current definition.")

    clutch["y"] = (clutch["scoreVal"].fillna(0) > 0).astype(int)

    # simple numeric features available in your schema
    feat_cols = ["clock_sec", "shotDistance", "xLegacy", "yLegacy", "shotVal", "isFieldGoal"]
    feat_cols = [c for c in feat_cols if c in clutch.columns]
    X = clutch[feat_cols].select_dtypes(include=["number"]).fillna(0)
    y = clutch["y"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=clutch["game_id"]))

    model = LogisticRegression(max_iter=200)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    proba = model.predict_proba(X.iloc[test_idx])[:, 1]
    auc = roc_auc_score(y.iloc[test_idx], proba)

    joblib.dump(model, MODEL_DIR / "clutch_classifier.pkl")

    out = clutch.iloc[test_idx][["game_id", "period", "clock_sec"]].copy()
    out["y_true"] = y.iloc[test_idx].values
    out["y_proba"] = proba
    out.to_csv(OUT_DIR / "clutch_predictions.csv", index=False)

    print("✅ Day 18 complete")
    print(f"AUC={auc:.4f}")
    print(classification_report(y.iloc[test_idx], (proba >= 0.5).astype(int)))

if __name__ == "__main__":
    main()
