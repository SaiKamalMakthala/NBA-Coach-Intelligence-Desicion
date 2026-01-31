import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

usage = pd.read_csv(PROJECT_ROOT / "data/processed/player_usage_spikes.csv")
defense = pd.read_csv(PROJECT_ROOT / "data/processed/player_defensive_workload.csv")
lineup = pd.read_csv(PROJECT_ROOT / "data/processed/player_lineup_stability.csv")

print("Files loaded successfully")

#Basic null & duplicate checks
#What this catches
#Missing feature values
#Broken joins
#Duplicate aggregation errors
def basic_checks(df, name):
    return {
        "table": name,
        "rows": len(df),
        "null_cells": df.isna().sum().sum(),
        "duplicate_rows": df.duplicated().sum()
    }

checks = []
checks.append(basic_checks(usage, "usage_spikes"))
checks.append(basic_checks(defense, "defensive_workload"))
checks.append(basic_checks(lineup, "lineup_stability"))

#Logical range checks
#Fatigue, workload, stability cannot be negative
#Prevents silent model corruption
logic_checks = []

logic_checks.append({
    "metric": "usage_spike_norm",
    "invalid_count": (usage["usage_spike_norm"] < -1).sum()
})

logic_checks.append({
    "metric": "defensive_workload_score",
    "invalid_count": (defense["defensive_workload_score"] < 0).sum()
})

logic_checks.append({
    "metric": "lineup_stability_score",
    "invalid_count": (lineup["lineup_stability_score"] <= 0).sum()
})

#Extreme value detection(outliers)
#These are game-breaking workloads
#Worth flagging, not removing
outliers = usage[
    usage["usage_spike_norm"] > usage["usage_spike_norm"].quantile(0.99)
]

print(f"High usage spike outliers: {len(outliers)}")

#Cross-feature consistency checks
#High offensive spike + low defensive load = plausible
#High on both = real fatigue risk
#Conflicts highlight suspicious data
merged = (
    usage
    .merge(defense, on=["game_id", "playerNameI", "period"], how="inner")
    .merge(lineup, on=["game_id", "playerNameI", "period"], how="inner")
)

merged["fatigue_conflict"] = (
    (merged["usage_spike_norm"] > 0.5) &
    (merged["defensive_workload_score"] < merged["defensive_workload_score"].median())
).astype(int)

#Save validation report
report = pd.DataFrame(checks)
report.to_csv(PROJECT_ROOT / "data/validation/sanity_report.csv", index=False)

print("Sanity checks completed and report saved")
