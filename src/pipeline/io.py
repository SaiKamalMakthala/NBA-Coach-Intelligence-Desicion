import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def load_csv(relative_path):
    """Load a CSV from the data directory."""
    path = DATA_DIR / relative_path
    return pd.read_csv(path)


def save_csv(df, relative_path):
    """Save a DataFrame to the data directory."""
    path = DATA_DIR / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
