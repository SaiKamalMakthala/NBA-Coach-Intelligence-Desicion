import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

pbp = pd.read_csv(DATA_DIR / "pbp_with_possessions.csv")
print(f"Loaded {len(pbp)} rows")

#Ensures fatigue accumulates in the correct order.

pbp = pbp.sort_values(
    ["game_id", "period", "clock_sec"],
    ascending=[True, True, False]
)

#Estimates how long a player was active via possession involvement.

player_minutes = (
    pbp.groupby(["game_id", "playerNameI", "period"])
       .agg(events_played=("possession_id", "count"))
       .reset_index()
)

player_minutes["minutes_proxy"] = player_minutes["events_played"] * 2.5 / 60

#Counts actions that are physically demanding.

pbp["high_intensity"] = (
    pbp["shotVal"].fillna(0) +
    pbp["isFieldGoal"].fillna(0)
)

intensity_load = (
    pbp.groupby(["game_id", "playerNameI"])
       .agg(high_intensity_events=("high_intensity", "sum"))
       .reset_index()
)

#Possesions without rest
#Detects long uninterrupted stretches.

pbp["possession_gap"] = (
    pbp.groupby(["game_id", "playerNameI"])["possession_id"]
       .diff()
       .fillna(0)
)

pbp["continuous_play"] = (pbp["possession_gap"] == 0).astype(int)

continuous_load = (
    pbp.groupby(["game_id", "playerNameI"])
       .agg(continuous_possessions=("continuous_play", "sum"))
       .reset_index()
)

#Combining fatigue proxies
#Creates a single fatigue feature table.

fatigue = (
    player_minutes
    .merge(intensity_load, on=["game_id", "playerNameI"], how="left")
    .merge(continuous_load, on=["game_id", "playerNameI"], how="left")
    .fillna(0)
)

#Creates a comparable fatigue metric.

fatigue["fatigue_score"] = (
    0.4 * fatigue["minutes_proxy"] +
    0.4 * fatigue["high_intensity_events"] +
    0.2 * fatigue["continuous_possessions"]
)
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "features"
OUTPUT_DIR.mkdir(exist_ok=True)

fatigue.to_csv(OUTPUT_DIR / "player_fatigue_v1.csv", index=False)
print("✅ Saved fatigue proxy features")
