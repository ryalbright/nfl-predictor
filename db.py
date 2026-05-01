"""
db.py — Load season_totals CSV into SQLite and expose query helpers.

Run directly to (re)build the database:
    python db.py
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH  = Path("nfl_stats.db")
CSV_PATH = Path("season_totals_2019_2025.csv")

SCHEMA = """
CREATE TABLE IF NOT EXISTS player_seasons (
    player_id       TEXT,
    player_name     TEXT,
    position        TEXT,
    team            TEXT,
    season          INTEGER,
    age_sept1       REAL,
    games           INTEGER,
    completions     REAL,
    attempts        REAL,
    passing_yards   REAL,
    passing_tds     REAL,
    interceptions   REAL,
    carries         REAL,
    rushing_yards   REAL,
    rushing_tds     REAL,
    targets         REAL,
    receptions      REAL,
    receiving_yards REAL,
    receiving_tds   REAL,
    ppr_points      REAL,
    scrimmage_yards REAL,
    PRIMARY KEY (player_id, season)
);
"""


def build_db(csv_path: Path = CSV_PATH, db_path: Path = DB_PATH) -> None:
    df = pd.read_csv(csv_path)

    # Keep the row with a valid age when there are duplicates
    if "age_sept1" in df.columns:
        df = df.sort_values("age_sept1", na_position="last")
    df = df.drop_duplicates(subset=["player_id", "season"], keep="first")

    cols = [
        "player_id", "player_name", "position", "team", "season",
        "age_sept1", "games",
        "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds",
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "ppr_points", "scrimmage_yards",
    ]
    df = df[[c for c in cols if c in df.columns]]

    con = sqlite3.connect(db_path)
    con.execute("DROP TABLE IF EXISTS player_seasons")
    con.execute(SCHEMA)
    df.to_sql("player_seasons", con, if_exists="append", index=False)
    con.commit()
    con.close()

    print(f"Built {db_path}  ({len(df):,} rows)")


def query(sql: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Run any SQL query and return a DataFrame."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(sql, con)
    con.close()
    return df


def get_seasons(
    positions: list[str] | None = None,
    min_games: int = 4,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Pull season rows filtered by position and minimum games played."""
    pos_clause = ""
    if positions:
        quoted = ", ".join(f"'{p}'" for p in positions)
        pos_clause = f"AND position IN ({quoted})"

    sql = f"""
        SELECT *
        FROM   player_seasons
        WHERE  games >= {min_games}
               {pos_clause}
        ORDER  BY player_id, season
    """
    return query(sql, db_path)


if __name__ == "__main__":
    build_db()

    # Quick sanity checks
    print()
    print(query("""
        SELECT position, COUNT(*) AS rows, ROUND(AVG(ppr_points), 1) AS avg_ppr
        FROM   player_seasons
        GROUP  BY position
        ORDER  BY avg_ppr DESC
    """))
