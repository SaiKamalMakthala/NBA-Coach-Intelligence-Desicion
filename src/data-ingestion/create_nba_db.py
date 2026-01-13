import sqlite3
import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = DATA_DIR / "db"
PROCESSED_DIR = DATA_DIR / "processed"

DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "nba_coach_intelligence.db"

#Connect to SQLite
#Opens (or creates) the SQLite database
#Establishes a persistent connection
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

#Create Tables (SQL)
cursor.execute("""
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    home TEXT,
    away TEXT,
    scoreHome INTEGER,
    scoreAway INTEGER
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS pbp (
    game_id TEXT,
    period INTEGER,
    clock_sec REAL,
    teamTricode TEXT,
    description TEXT,
    possession_id INTEGER
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS possessions (
    game_id TEXT,
    possession_id INTEGER,
    teamTricode TEXT,
    points INTEGER,
    duration_sec REAL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS lineup_stints (
    game_id TEXT,
    teamTricode TEXT,
    stint_id INTEGER,
    stint_duration_sec REAL
)
""")

conn.commit()

#Load CSVs into pandas
pbp = pd.read_csv(PROCESSED_DIR / "pbp_with_possessions.csv")
lineups = pd.read_csv(PROCESSED_DIR / "lineup_stints.csv")

print("Loaded data:")
print(f"PBP rows: {len(pbp):,}")
print(f"Lineup rows: {len(lineups):,}")

#Insert Data into Database
#Transfers processed data → SQL tables
#Preserves analytics-ready structure
pbp[[
    "game_id",
    "period",
    "clock_sec",
    "teamTricode",
    "description",
    "possession_id"
]].to_sql(
    "pbp",
    conn,
    if_exists="append",
    index=False
)

lineups[[
    "game_id",
    "teamTricode",
    "stint_id",
    "stint_duration_sec"
]].to_sql(
    "lineup_stints",
    conn,
    if_exists="append",
    index=False
)

#Sanity Check with SQL
query = """
SELECT teamTricode,
       COUNT(*) AS num_stints,
       AVG(stint_duration_sec) AS avg_stint
FROM lineup_stints
GROUP BY teamTricode
LIMIT 5;
"""

print(pd.read_sql(query, conn))

conn.close()
print("✅ NBA database created successfully")
