import pandas as pd
from src.features.config import PROCESSED_DIR, FEATURES_DIR, KEYS
#Loads feature files safely
#Ensures every dataset can be joined without silent bugs
def load_feature_table(filename: str) -> pd.DataFrame:
    """Loads a processed feature CSV and guarantees consistent join keys."""
    df = pd.read_csv(PROCESSED_DIR / filename)
    missing = [k for k in KEYS if k not in df.columns]
    if missing:
        raise ValueError(f"{filename} missing keys: {missing}")
    return df

#Load all feature outputs (Day 8–13)
def load_all_features():
    usage = load_feature_table("player_usage_spikes.csv")
    defense = load_feature_table("player_defensive_workload.csv")
    lineup = load_feature_table("player_lineup_stability.csv")

    # Optional if you saved fatigue output in processed/
    fatigue_path = PROCESSED_DIR / "player_fatigue_v1.csv"
    fatigue = pd.read_csv(fatigue_path) if fatigue_path.exists() else None

    return usage, defense, lineup, fatigue

#Standardize column names & avoid collisions
def prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Prefix non-key columns to avoid name collisions."""
    rename_map = {
        c: f"{prefix}__{c}"
        for c in df.columns
        if c not in KEYS
    }
    return df.rename(columns=rename_map)

#Merge into one feature mart (CORE STEP)
def build_feature_mart():
    usage, defense, lineup, fatigue = load_all_features()

    usage = prefix_columns(usage, "usage")
    defense = prefix_columns(defense, "defense")
    lineup = prefix_columns(lineup, "lineup")

    mart = usage.merge(defense, on=KEYS, how="inner").merge(lineup, on=KEYS, how="inner")

    if fatigue is not None:
        # Ensure fatigue has keys; if it doesn't include period, we can merge at (game, player) later
        if all(k in fatigue.columns for k in KEYS):
            fatigue = prefix_columns(fatigue, "fatigue")
            mart = mart.merge(fatigue, on=KEYS, how="left")
        else:
            print("⚠️ fatigue table missing period-level keys; skipping period merge")

    return mart

def finalize_and_save(mart: pd.DataFrame):
    # Fill remaining NaNs (safe defaults)
    mart = mart.fillna(0)

    out_path = FEATURES_DIR / "feature_mart.csv"
    mart.to_csv(out_path, index=False)

    print(f"✅ Saved feature mart: {out_path}")
    print(f"Rows: {len(mart):,} | Columns: {mart.shape[1]}")
#Add the entry point
def main():
    mart = build_feature_mart()
    finalize_and_save(mart)

if __name__ == "__main__":
    main()
