# src/models/explain_with_shap.py
import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = PROJECT_ROOT / "data/features/feature_mart.csv"
MODEL_PATH = PROJECT_ROOT / "data/models/fatigue_model.pkl"
OUT_DIR = PROJECT_ROOT / "data/reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["game_id", "playerNameI", "period"]

def main():
    try:
        import shap
    except ImportError:
        raise SystemExit("Install shap first: pip install shap")

    df = pd.read_csv(FEATURE_PATH)

    # Build X the same way as training: drop keys, keep numeric, fillna(0)
    X = df.drop(columns=[c for c in KEYS if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=["number"]).fillna(0)

    model = joblib.load(MODEL_PATH)

    # ✅ Ensure exact feature alignment if model stores feature names (sklearn >=1.0)
    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    # Use a background sample for speed
    background = X.sample(min(1000, len(X)), random_state=42)
    sample = X.sample(min(300, len(X)), random_state=7)

    # ✅ Use TreeExplainer explicitly with additivity check disabled
    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    shap_values = explainer.shap_values(sample, check_additivity=False)

    # Compute global importance = mean(|SHAP|)
    import numpy as np
    mean_abs = np.abs(shap_values).mean(axis=0)

    feat_imp = (
        pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
    )
    feat_imp.to_csv(OUT_DIR / "shap_top_features.csv", index=False)

    print("✅ Day 20 complete: data/reports/shap_top_features.csv")

if __name__ == "__main__":
    main()
