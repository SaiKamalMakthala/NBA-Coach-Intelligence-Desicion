import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PBP_PATH = PROCESSED_DIR / "pbp_with_possessions.csv"
OUTPUT_PATH = PROCESSED_DIR / "shot_features.csv"
#Load PBP data
pbp = pd.read_csv(PBP_PATH)
print(f"Loaded {len(pbp):,} total PBP rows")
#Isolate shot attempts(Boolean filter One row = one shot attempt)
shots = pbp[pbp["isFieldGoal"] == 1].copy()
print(f"Filtered {len(shots):,} shot attempts")
#Shot outcome(2pt/3pt)
shots["made_shot"] = (shots["scoreVal"] > 0).astype(int)
shots["shot_type"] = shots["shotVal"].map({2: "2PT", 3: "3PT"})
#Time pressure feature
shots["late_clock"] = (shots["clock_sec"] <= 5).astype(int)
shots["early_clock"] = (shots["clock_sec"] >= 18).astype(int)
#Final Shot features Column
shot_features = shots[
    [
        "game_id",
        "period",
        "possession_id",
        "teamTricode",
        "playerNameI",
        "shot_type",
        "shotDistance",
        "xLegacy",
        "yLegacy",
        "clock_sec",
        "early_clock",
        "late_clock",
        "made_shot",
        "scoreVal",
    ]
]
shot_features.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Saved shot feature table to {OUTPUT_PATH}")
print(f"Total shots: {len(shot_features):,}")
