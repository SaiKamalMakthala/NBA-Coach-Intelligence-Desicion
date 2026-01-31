# src/models/train_game_control_index.py
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PBP_PATH = PROJECT_ROOT / "data/processed/pbp_with_possessions.csv"
RUNS_PATH = PROJECT_ROOT / "data/processed/team_momentum_runs.csv"
OUT_PATH = PROJECT_ROOT / "data/outputs/game_control_index.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def main():
    pbp = pd.read_csv(PBP_PATH)
    runs = pd.read_csv(RUNS_PATH)

    # period points per team (approx): sum(scoreVal) by teamTricode, period
    pbp["points"] = pbp["scoreVal"].fillna(0)
    team_period = (
        pbp.groupby(["game_id", "period", "teamTricode"], as_index=False)
           .agg(period_points=("points", "sum"),
                possessions=("possession_id", "nunique"))
    )

    # major runs per team per game (runs table has team + is_major_run)
    if "is_major_run" not in runs.columns:
        # if your run file uses different flag name, adjust here
        runs["is_major_run"] = (runs["points_scored"] >= 8).astype(int)

    runs_team = (
        runs.groupby(["game_id", "team"], as_index=False)
            .agg(major_runs=("is_major_run", "sum"),
                 run_points=("points_scored", "sum"))
            .rename(columns={"team": "teamTricode"})
    )

    team_period = team_period.merge(runs_team, on=["game_id", "teamTricode"], how="left").fillna(0)

    # Game Control Index (simple interpretable v1)
    team_period["pts_per_poss"] = team_period["period_points"] / (team_period["possessions"] + 1)
    team_period["game_control_index"] = (
        0.6 * team_period["pts_per_poss"] +
        0.3 * team_period["major_runs"] +
        0.1 * (team_period["run_points"] / 100.0)
    )

    team_period.to_csv(OUT_PATH, index=False)
    print(f"✅ Day 16 complete: {OUT_PATH}")

if __name__ == "__main__":
    main()
