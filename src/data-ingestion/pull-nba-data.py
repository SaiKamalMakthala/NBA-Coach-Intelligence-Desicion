from pathlib import Path
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

# =============================
# CONFIG
# =============================
SEASON = "2023-24"
SEASON_ID_MAP = {"2023-24": "22023"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Offline PBP file (downloaded once)
PBP_OFFLINE_PATH = RAW_DATA_DIR / "nba_pbp_2023_24.csv"

# =============================
# DIRECTORY SAFETY
# =============================
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# =============================
# 1️⃣ FETCH NBA GAMES METADATA
# =============================
print("📥 Fetching NBA games metadata...")

games = leaguegamefinder.LeagueGameFinder(
    season_nullable=SEASON,
    league_id_nullable="00"
)

games_df = games.get_data_frames()[0]

# Filter correct season
games_df = games_df[games_df["SEASON_ID"] == SEASON_ID_MAP[SEASON]]

# Standardize column name
games_df.rename(columns={"GAME_ID": "game_id"}, inplace=True)

# Save games
games_path = RAW_DATA_DIR / f"nba_games_{SEASON}.csv"
games_df.to_csv(games_path, index=False)

print(f"✅ Saved {len(games_df)} games to {games_path}")

# =============================
# 2️⃣ LOAD OFFLINE PLAY-BY-PLAY
# =============================
print("📥 Loading offline play-by-play data...")

if not PBP_OFFLINE_PATH.exists():
    raise FileNotFoundError(
        f"❌ Missing PBP file: {PBP_OFFLINE_PATH}\n"
        f"Download 2023–24 NBA PBP CSV and place it in data/raw/"
    )

pbp_df = pd.read_csv(PBP_OFFLINE_PATH)
print(f"✅ Loaded {len(pbp_df):,} raw PBP rows")

# =============================
# 3️⃣ ENSURE PBP HAS game_id
# =============================
if "game_id" not in pbp_df.columns:
    raise ValueError(
        f"❌ Expected 'game_id' column not found in PBP data.\n"
        f"Available columns: {pbp_df.columns.tolist()}"
    )

# =============================
# 4️⃣ FILTER PBP TO 2023–24 GAMES
# =============================
season_game_ids = set(games_df["game_id"].unique())

pbp_df = pbp_df[pbp_df["game_id"].isin(season_game_ids)]

print(f"🎯 Filtered to {len(pbp_df):,} PBP rows for season {SEASON}")

# =============================
# 5️⃣ SAVE FILTERED PBP
# =============================
pbp_out_path = RAW_DATA_DIR / "nba_pbp_2023_24_filtered.csv"
pbp_df.to_csv(pbp_out_path, index=False)

print(f"✅ Saved filtered PBP to {pbp_out_path}")
print("🎉 NBA games + play-by-play ingestion complete!")
