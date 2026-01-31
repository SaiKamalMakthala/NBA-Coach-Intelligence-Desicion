# src/decision_support/game_briefs.py
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "data/reports/game_briefs.md"
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    insights_path = PROJECT_ROOT / "data/outputs/coach_insights.csv"
    if not insights_path.exists():
        raise FileNotFoundError("Run Day 22 first to generate coach_insights.csv")

    ins = pd.read_csv(insights_path)
    lines = ["# Game Briefs\n"]
    for game_id, g in ins.groupby("game_id"):
        lines.append(f"## {game_id}\n")
        for _, r in g.head(8).iterrows():
            lines.append(f"- [{r['type']}] {r['insight']}\n")
        lines.append("\n")

    OUT.write_text("".join(lines))
    print("✅ Day 25 complete: data/reports/game_briefs.md")

if __name__ == "__main__":
    main()
