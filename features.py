"""
features.py — Build training/prediction feature sets from the SQLite database.

Features include:
  - Current season stats (rates + efficiency)
  - 2-season lag history (lag1, lag2 PPR totals, trend, peak)
  - Prior positional rank (were they top-6/12 last year?)
  - Team change flag (role disruption signal)
  - Durability/injury features (games played history)
"""

import numpy as np
import pandas as pd
from db import get_seasons

POSITIONS = ["QB", "RB", "WR", "TE"]

INJURY_GAMES_THRESHOLD = 11   # missed 4+ of 17 games = injury-shortened
HOLDOUT_SEASON         = 2024  # 2024→2025 is held out for honest evaluation

POS_FEATURES: dict[str, list[str]] = {
    "QB": [
        # Identity / health
        "age_sept1", "games_pct", "prev_games_pct", "career_durability", "injury_last2",
        # Current season volume
        "attempts", "completions", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds",
        # Current season rates
        "ppr_per_game", "pass_yds_per_game", "rush_yds_per_game",
        "completion_rate", "td_int_ratio",
        # 2-year history
        "lag1_ppr", "lag2_ppr", "ppr_trend", "peak_ppr_2yr",
        "lag1_ppr_per_game", "lag1_top6", "lag1_top12",
        "team_changed",
    ],
    "RB": [
        "age_sept1", "games_pct", "prev_games_pct", "career_durability", "injury_last2",
        "carries", "rushing_yards", "rushing_tds",
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "ppr_per_game", "rush_yds_per_game", "yards_per_carry", "targets_per_game",
        "lag1_ppr", "lag2_ppr", "ppr_trend", "peak_ppr_2yr",
        "lag1_ppr_per_game", "lag1_top6", "lag1_top12",
        "team_changed",
    ],
    "WR": [
        "age_sept1", "games_pct", "prev_games_pct", "career_durability", "injury_last2",
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "carries", "rushing_yards",
        "ppr_per_game", "rec_yds_per_game", "yards_per_target", "catch_rate",
        "lag1_ppr", "lag2_ppr", "ppr_trend", "peak_ppr_2yr",
        "lag1_ppr_per_game", "lag1_top6", "lag1_top12",
        "team_changed",
    ],
    "TE": [
        "age_sept1", "games_pct", "prev_games_pct", "career_durability", "injury_last2",
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "ppr_per_game", "rec_yds_per_game", "yards_per_target", "catch_rate",
        "lag1_ppr", "lag2_ppr", "ppr_trend", "peak_ppr_2yr",
        "lag1_ppr_per_game", "lag1_top6", "lag1_top12",
        "team_changed",
    ],
}


# ── Feature computation ────────────────────────────────────────────────────────

def add_durability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Injury and durability signals — needs full career history per player."""
    df = df.copy().sort_values(["player_id", "season"]).reset_index(drop=True)

    df["games_pct"] = (df["games"] / 17).clip(0, 1)
    df["prev_games_pct"] = df.groupby("player_id")["games_pct"].shift(1)

    missed_this = df["games"] <= INJURY_GAMES_THRESHOLD
    missed_prev = df.groupby("player_id")["games"].shift(1) <= INJURY_GAMES_THRESHOLD
    df["injury_last2"] = (missed_this | missed_prev.fillna(False)).astype(float)

    df["career_durability"] = (
        df.groupby("player_id")["games_pct"]
          .transform(lambda s: s.shift(1).expanding().mean())
    )
    return df


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    2-season lag PPR, trend, positional rank history, and team change.
    Must be called on the full position-filtered DataFrame so ranks are correct.
    """
    df = df.copy().sort_values(["player_id", "season"]).reset_index(drop=True)

    # ── Lag PPR totals ──────────────────────────────────────────────────
    df["lag1_ppr"] = df.groupby("player_id")["ppr_points"].shift(1)
    df["lag2_ppr"] = df.groupby("player_id")["ppr_points"].shift(2)

    # Year-over-year trend: positive = player is improving heading into this season
    df["ppr_trend"] = df["ppr_points"] - df["lag1_ppr"]

    # Best season over the past 2 years — captures ceiling even after an injury year
    df["peak_ppr_2yr"] = df[["lag1_ppr", "lag2_ppr"]].max(axis=1)

    # Prior year per-game rate (useful when current season was injury-shortened)
    lag1_games = df.groupby("player_id")["games"].shift(1).replace(0, np.nan)
    df["lag1_ppr_per_game"] = df["lag1_ppr"] / lag1_games

    # ── Prior positional rank ───────────────────────────────────────────
    # Rank within each season (1 = best), then shift forward one year
    df["_curr_rank"] = df.groupby("season")["ppr_points"].rank(
        ascending=False, method="min"
    )
    df["lag1_rank"] = df.groupby("player_id")["_curr_rank"].shift(1)
    df["lag1_top6"]  = (df["lag1_rank"] <= 6).astype(float)
    df["lag1_top12"] = (df["lag1_rank"] <= 12).astype(float)
    df["lag1_top24"] = (df["lag1_rank"] <= 24).astype(float)
    df = df.drop(columns=["_curr_rank"])

    # ── Team change ─────────────────────────────────────────────────────
    prev_team = df.groupby("player_id")["team"].shift(1)
    df["team_changed"] = (df["team"] != prev_team).astype(float)
    # First season for a player: no prior team → not a "change"
    df.loc[prev_team.isna(), "team_changed"] = 0.0

    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Per-game rates and efficiency — row-level, no career context needed."""
    df = df.copy()
    g = df["games"].replace(0, np.nan)

    df["ppr_per_game"]      = df["ppr_points"]      / g
    df["pass_yds_per_game"] = df["passing_yards"]   / g
    df["rush_yds_per_game"] = df["rushing_yards"]   / g
    df["rec_yds_per_game"]  = df["receiving_yards"] / g
    df["targets_per_game"]  = df["targets"]          / g

    df["completion_rate"]  = df["completions"]     / df["attempts"].replace(0, np.nan)
    df["td_int_ratio"]     = df["passing_tds"]     / (df["interceptions"] + 1)
    df["yards_per_carry"]  = df["rushing_yards"]   / df["carries"].replace(0, np.nan)
    df["yards_per_target"] = df["receiving_yards"] / df["targets"].replace(0, np.nan)
    df["catch_rate"]       = df["receptions"]      / df["targets"].replace(0, np.nan)

    return df


def _load_full(position: str) -> pd.DataFrame:
    """Load all seasons for a position with every feature layer applied."""
    df = get_seasons(positions=[position], min_games=1)
    df = add_durability_features(df)
    df = add_history_features(df)   # ← new: lag PPR, rank, team change
    df = engineer(df)
    return df


# ── Dataset builders ───────────────────────────────────────────────────────────

def build_training_data(
    position: str,
    min_games: int = 4,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Year-N features → year-N+1 PPR.
    Holdout (2024→2025) is excluded so evaluation is honest.
    """
    df = _load_full(position)
    df = df[df["games"] >= min_games]

    next_yr = (
        df[["player_id", "season", "ppr_points"]]
        .copy()
        .rename(columns={"ppr_points": "next_ppr"})
    )
    next_yr["season"] = next_yr["season"] - 1

    merged = df.merge(next_yr, on=["player_id", "season"], how="inner")
    merged = merged[merged["season"] < HOLDOUT_SEASON]   # strict holdout

    feature_cols = [c for c in POS_FEATURES[position] if c in merged.columns]
    X = merged[feature_cols].fillna(0)
    y = merged["next_ppr"]
    return X, y, feature_cols


def build_predict_data(
    position: str,
    season: int,
    feature_cols: list[str],
    min_games: int = 1,
) -> pd.DataFrame:
    """Full feature set for `season` — ready for model.predict()."""
    df = _load_full(position)
    df = df[(df["season"] == season) & (df["games"] >= min_games)]
    return df.reset_index(drop=True)


if __name__ == "__main__":
    print("Feature counts and sample sizes after adding 2-year history:\n")
    for pos in POSITIONS:
        X, y, cols = build_training_data(pos)
        new = [c for c in cols if c in (
            "lag1_ppr", "lag2_ppr", "ppr_trend", "peak_ppr_2yr",
            "lag1_top6", "lag1_top12", "team_changed",
        )]
        print(f"  {pos}: {len(X)} training pairs | {len(cols)} features")
        print(f"       new history features present: {new}\n")
