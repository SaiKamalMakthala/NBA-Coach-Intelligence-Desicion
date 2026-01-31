from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["game_id", "playerNameI", "period"]
