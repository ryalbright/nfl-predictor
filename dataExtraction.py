# dataExtraction.py
# Build season totals (2019–2025) with PPR, snaps, routes, YPRR.
# Fetches directly from nflverse-data GitHub releases (works with pandas 3.x).

import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

SEASONS = list(range(2019, 2026))
POS_KEEP = {"QB", "RB", "WR", "TE"}
MIN_ROUTES_FOR_YPRR = 50

# nflverse GitHub release URLs
WEEKLY_URL  = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.parquet"
ROSTERS_URL = "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{year}.parquet"

# ---------- helpers ----------
def age_on_sept1(birth_date, year):
    if pd.isna(birth_date): return pd.NA
    b = pd.to_datetime(birth_date)
    r = relativedelta(datetime(int(year), 9, 1), b)
    return round(r.years + r.months / 12 + r.days / 365.25, 2)

def mode_or_last(s):
    m = s.mode()
    return m.iat[0] if len(m) else s.iloc[-1]

def compute_ppr(r):
    return (
        (r.get("receptions") or 0) * 1.0
        + ((r.get("receiving_yards") or 0) + (r.get("rushing_yards") or 0)) * 0.1
        + (r.get("passing_yards") or 0) * 0.04
        + ((r.get("receiving_tds") or 0) + (r.get("rushing_tds") or 0)) * 6.0
        + (r.get("passing_tds") or 0) * 4.0
        - (r.get("interceptions") or 0) * 2.0
        + (r.get("two_point_conversions") or 0) * 2.0
        - (r.get("fumbles_lost") or 0) * 2.0
    )

# ---------- load weekly data ----------
print("Fetching weekly stats...")
frames = []
for yr in SEASONS:
    url = WEEKLY_URL.format(year=yr)
    try:
        df = pd.read_parquet(url, engine="pyarrow")
        df["season"] = yr
        frames.append(df)
        print(f"  {yr}: {len(df):,} rows")
    except Exception as e:
        print(f"  {yr}: skipped ({e})")

weekly = pd.concat(frames, ignore_index=True)

# regular season only — playoffs inflate counting stats
if "season_type" in weekly.columns:
    weekly = weekly[weekly["season_type"] == "REG"].copy()

# normalize common column name variants
weekly = weekly.rename(columns={
    "player_id":              "player_id",
    "gsis_id":                "player_id",
    "player_name":            "player_name",
    "display_name":           "player_name",
    "position":               "position",
    "recent_team":            "team",
    "fantasy_points_ppr":     "ppr_points",
    "passing_interceptions":  "interceptions",   # new nflverse column name
})

# filter positions
if "position" not in weekly.columns and "pos" in weekly.columns:
    weekly = weekly.rename(columns={"pos": "position"})
weekly = weekly[weekly["position"].isin(POS_KEEP)].copy()

# add PPR if not already present
if "ppr_points" not in weekly.columns:
    weekly["ppr_points"] = weekly.apply(compute_ppr, axis=1)

# aggregate to season
sum_cols = [
    "attempts", "completions", "passing_yards", "passing_tds", "interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "fumbles_lost", "two_point_conversions", "ppr_points",
]
agg = {c: "sum" for c in sum_cols if c in weekly.columns}
if "team" in weekly.columns:
    agg["team"] = mode_or_last

group_cols = [c for c in ["player_id", "player_name", "position", "season"] if c in weekly.columns]
seasonal = weekly.groupby(group_cols, as_index=False).agg(agg)

# games played
week_col = next((c for c in ["week", "game_week"] if c in weekly.columns), None)
if week_col:
    games = (
        weekly.groupby(["player_id", "season"], as_index=False)[week_col]
              .nunique()
              .rename(columns={week_col: "games"})
    )
    seasonal = seasonal.merge(games, on=["player_id", "season"], how="left")

# scrimmage yards
rush = seasonal.get("rushing_yards", 0)
recv = seasonal.get("receiving_yards", 0)
seasonal["scrimmage_yards"] = rush.fillna(0) + recv.fillna(0)

# load rosters to get ages
print("\nFetching rosters for age calculation...")
roster_frames = []
for yr in SEASONS:
    url = ROSTERS_URL.format(year=yr)
    try:
        r = pd.read_parquet(url, engine="pyarrow")
        roster_frames.append(r)
    except Exception:
        pass  # roster data is optional

if roster_frames:
    rosters = pd.concat(roster_frames, ignore_index=True)
    id_col = next((c for c in ["player_id", "gsis_id"] if c in rosters.columns), None)
    if id_col and id_col != "player_id":
        rosters = rosters.rename(columns={id_col: "player_id"})
    if "player_id" in rosters.columns and "birth_date" in rosters.columns:
        birth = rosters[["player_id", "birth_date"]].drop_duplicates("player_id")
        seasonal = seasonal.merge(birth, on="player_id", how="left")
        seasonal["age_sept1"] = seasonal.apply(
            lambda r: age_on_sept1(r["birth_date"], r["season"]), axis=1
        )

# trim to only essential data
essentials = [
    "player_id", "player_name", "position", "team", "season", "age_sept1", "games",
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "ppr_points", "scrimmage_yards",
]
seasonal = seasonal[[c for c in essentials if c in seasonal.columns]]

# write
out = "season_totals_2019_2025.csv"
seasonal.to_csv(out, index=False)
print(f"\nWrote: {out}  ({len(seasonal):,} rows, seasons {SEASONS[0]}–{SEASONS[-1]})")
