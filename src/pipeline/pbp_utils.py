import pandas as pd


def add_shot_flags(pbp):
    """Identify shot attempts and makes."""
    pbp["is_shot"] = pbp["isFieldGoal"].astype(int)
    pbp["is_make"] = (pbp["shotVal"] > 0).astype(int)
    return pbp


def add_possession_end(pbp):
    """Flag possession-ending events."""
    pbp["ends_possession"] = (
        pbp["actionType"].isin(["Made Shot", "Missed Shot", "Turnover"])
    ).astype(int)
    return pbp


def compute_possession_metrics(pbp):
    """Aggregate possession-level metrics."""
    possession_metrics = (
        pbp.groupby(["game_id", "possession_id", "teamTricode"])
        .agg(
            points=("scoreVal", "sum"),
            duration_sec=("clock_sec", lambda x: x.max() - x.min()),
            shots=("is_shot", "sum"),
            makes=("is_make", "sum"),
        )
        .reset_index()
    )
    return possession_metrics
