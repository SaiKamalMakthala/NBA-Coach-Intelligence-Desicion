import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PBP_PATH = PROCESSED_DIR / "pbp_with_possessions.csv"

pbp = pd.read_csv(PBP_PATH)
print(f"Loaded {len(pbp):,} PBP rows")
#Identifying Substitution Events (Text Filtering)
#Substitutions are stored in text description not numeric flags
subs = pbp[pbp["description"].str.contains("SUB", na=False)].copy()
print(f"Found {len(subs):,} substitution events")
#Parse Player IN / Player OUT (Text Extraction)
#Extracts player leaving the floor
#Extracts player entering the floor
#Converts raw text → structured data
subs["player_out"] = subs["description"].str.extract(r"SUB:\s(.*?)\sFOR")
subs["player_in"] = subs["description"].str.extract(r"FOR\s(.*)")
#Creating Substitution Timestamp
subs["game_time_sec"] = (
    (subs["period"] - 1) * 720
    + (720 - subs["clock_sec"])
)
#Saving substitution table
subs_output = PROCESSED_DIR / "substitutions.csv"
subs[
    [
        "game_id",
        "teamTricode",
        "player_out",
        "player_in",
        "period",
        "clock_sec",
        "game_time_sec"
    ]
].to_csv(subs_output, index=False)

print(f"✅ Saved substitutions to {subs_output}")
#Build Lineup Stints (Conceptual Core)
#Orders substitutions chronologically
#Creates a unique lineup stint ID
#Each stint represents a lineup change
subs = subs.sort_values(
    ["game_id", "teamTricode", "game_time_sec"]
)

subs["stint_id"] = (
    subs
    .groupby(["game_id", "teamTricode"])
    .cumcount()
)
#Estimate Lineup Stint Duration
#Calculates how long each lineup stayed on court
#Enables lineup stability & fatigue analysis
subs["stint_start"] = subs["game_time_sec"]
subs["stint_end"] = subs.groupby(
    ["game_id", "teamTricode"]
)["game_time_sec"].shift(-1)

subs["stint_duration_sec"] = subs["stint_end"] - subs["stint_start"]
#Saving linup stints
lineup_output = PROCESSED_DIR / "lineup_stints.csv"

subs[
    [
        "game_id",
        "teamTricode",
        "stint_id",
        "player_out",
        "player_in",
        "stint_start",
        "stint_end",
        "stint_duration_sec"
    ]
].to_csv(lineup_output, index=False)

print(f"✅ Saved lineup stints to {lineup_output}")
