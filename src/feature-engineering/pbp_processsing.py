import pandas as pd
from pathlib import Path
# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PBP_PATH = PROCESSED_DIR / "pbp_with_possessions.csv"
OUTPUT_PATH = PROCESSED_DIR / "possession_level_metrics.csv"
#Load Play-by-Play with Possessions
pbp = pd.read_csv(PBP_PATH)
print(f"Loaded {len(pbp):,} play-by-play rows")
pbp.head()
#Create core event flag(Analytics logic)
pbp["is_shot"] = pbp["isFieldGoal"].astype(int)
pbp["points_scored"] = pbp["scoreVal"].fillna(0)


#Possesion level grouping
#Converts plays to possesion
possession_df = (
    pbp
    .groupby(
        ["game_id", "period", "possession_id", "teamTricode"],
        as_index=False
    )
    .agg(
        possession_points=("points_scored", "sum"),
        shot_attempts=("is_shot", "sum"),
        events_in_possession=("clock", "count"),
        possession_duration_sec=("clock_sec", "max")
    )
)

#Adding Possession Outcome Labels
#Adds basketball meaning to data
#Enables clutch / efficiency analysis later
def possession_outcome(row):
    if row["possession_points"] > 0:
        return "Score"
    elif row["shot_attempts"] > 0:
        return "Miss"
    else:
        return "Empty"

possession_df["possession_outcome"] = possession_df.apply(
    possession_outcome, axis=1
)

possession_df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Saved possession-level metrics to {OUTPUT_PATH}")
print(f"Total possessions: {len(possession_df):,}")
