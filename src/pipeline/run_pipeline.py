from .io import load_csv, save_csv
from .pbp_utils import (
    add_shot_flags,
    add_possession_end,
    compute_possession_metrics,
)


def main():
    print("🚀 Running NBA analytics pipeline...")

    # Load cleaned PBP
    pbp = load_csv("processed/pbp_with_possessions.csv")
    print(f"Loaded {len(pbp):,} PBP rows")

    # Feature engineering
    pbp = add_shot_flags(pbp)
    pbp = add_possession_end(pbp)

    # Save enriched PBP
    save_csv(pbp, "processed/pbp_enriched.csv")
    print("✅ Saved enriched PBP")

    # Possession-level table
    possessions = compute_possession_metrics(pbp)
    save_csv(possessions, "processed/possession_metrics.csv")
    print("✅ Saved possession metrics")

    print("🎉 Pipeline completed successfully")


if __name__ == "__main__":
    main()
