"""
classify.py — Predict probability of finishing top 6 / 12 / 24 at position in 2026.

Uses GradientBoostingClassifier with probability calibration so the percentages
are trustworthy (70% really means ~70% of the time).

Usage:
    python classify.py          # train all models then predict
    python classify.py --predict-only
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from db import get_seasons
from features import POSITIONS, POS_FEATURES, engineer, _load_full, HOLDOUT_SEASON

THRESHOLDS  = [6, 12, 24]
MODEL_DIR   = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ── Data builders ──────────────────────────────────────────────────────────────

def _rank_labels(position: str, top_n: int, min_games_rank: int = 1) -> pd.DataFrame:
    """
    For every season, rank all players by PPR and return a binary label:
      1 = finished top-N at position that season, 0 = did not.
    """
    all_seasons = get_seasons(positions=[position], min_games=min_games_rank)
    frames = []
    for season, grp in all_seasons.groupby("season"):
        grp = grp.copy()
        grp["rank"]  = grp["ppr_points"].rank(ascending=False, method="min")
        grp["label"] = (grp["rank"] <= top_n).astype(int)
        frames.append(grp[["player_id", "season", "label"]])
    return pd.concat(frames, ignore_index=True)


def build_classification_data(
    position: str,
    top_n: int,
    min_games: int = 4,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Return (X, y, feature_cols) where:
      X = year-N features for players who played in year N+1
      y = 1 if they finished top-N in year N+1, 0 otherwise
    """
    df = _load_full(position)
    df = df[df["games"] >= min_games]

    labels = _rank_labels(position, top_n)
    labels = labels.rename(columns={"season": "next_season", "label": "label"})
    labels["season"] = labels["next_season"] - 1

    merged = df.merge(labels[["player_id", "season", "label"]], on=["player_id", "season"])

    # Exclude holdout pair — same boundary as the regression model
    merged = merged[merged["season"] < HOLDOUT_SEASON]

    feature_cols = [c for c in POS_FEATURES[position] if c in merged.columns]
    X = merged[feature_cols].fillna(0)
    y = merged["label"]
    return X, y, feature_cols


def build_predict_features(position: str, season: int, feature_cols: list[str], min_games: int = 6) -> pd.DataFrame:
    """Return engineered + durability features for players with enough games in `season`."""
    df = _load_full(position)
    df = df[(df["season"] == season) & (df["games"] >= min_games)]
    return df.reset_index(drop=True)


# ── Training ───────────────────────────────────────────────────────────────────

def train_classifier(position: str, top_n: int) -> dict | None:
    X, y, feature_cols = build_classification_data(position, top_n)

    pos_count = y.sum()
    if pos_count < 5 or (len(y) - pos_count) < 5:
        print(f"  [{position} top-{top_n}] Not enough positive examples ({int(pos_count)}), skipping.")
        return None

    base = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    # CalibratedClassifierCV wraps the classifier so predict_proba() is reliable
    model = CalibratedClassifierCV(base, cv=5, method="isotonic")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    model.fit(X, y)
    probs      = model.predict_proba(X)[:, 1]
    brier      = brier_score_loss(y, probs)

    key = f"{position.lower()}_top{top_n}_clf"
    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_DIR / f"{key}.pkl")

    print(
        f"  [{position} top-{top_n:2d}]  "
        f"n={len(X)}  pos={int(pos_count)}  "
        f"CV AUC={auc_scores.mean():.3f}  Brier={brier:.3f}"
    )
    return {"model": model, "feature_cols": feature_cols}


def train_all() -> None:
    print("Training classifiers...\n")
    for pos in POSITIONS:
        for n in THRESHOLDS:
            train_classifier(pos, n)
        print()


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_position(position: str, season: int = 2025, top_n_display: int = 20) -> pd.DataFrame:
    # Load all three threshold models for this position
    models = {}
    for n in THRESHOLDS:
        path = MODEL_DIR / f"{position.lower()}_top{n}_clf.pkl"
        if path.exists():
            bundle      = joblib.load(path)
            models[n]   = bundle

    if not models:
        print(f"  [{position}] No classifier models found — run without --predict-only first.")
        return pd.DataFrame()

    df = build_predict_features(position, season, feature_cols=[])
    if df.empty:
        return pd.DataFrame()

    result = df[["player_name", "team", "age_sept1", "games", "ppr_points"]].copy()
    result = result.rename(columns={
        "player_name": "Player",
        "team":        "Team",
        "age_sept1":   "Age",
        "games":       "GP",
        "ppr_points":  f"{season} PPR",
    })

    for n, bundle in models.items():
        model        = bundle["model"]
        feature_cols = bundle["feature_cols"]
        X = df.reindex(columns=feature_cols).fillna(0)
        probs = model.predict_proba(X)[:, 1]
        result[f"Top-{n} %"] = (probs * 100).round(1)

    # sort by Top-12 if available, else Top-24
    sort_col = "Top-12 %" if "Top-12 %" in result.columns else f"Top-{THRESHOLDS[-1]} %"
    result = (
        result.sort_values(sort_col, ascending=False)
              .head(top_n_display)
              .reset_index(drop=True)
    )
    result.index += 1
    return result


def predict_all(season: int = 2025, top_n_display: int = 20) -> None:
    print(f"\n{'=' * 60}")
    print(f"  2026 Season Finish Probabilities  (based on {season} stats)")
    print(f"{'=' * 60}")
    for pos in POSITIONS:
        print(f"\n── {pos} ──")
        result = predict_position(pos, season=season, top_n_display=top_n_display)
        if not result.empty:
            print(result.to_string())


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFL finish-probability classifier.")
    parser.add_argument("--predict-only", action="store_true", help="Skip training, load saved models.")
    parser.add_argument("--season",       type=int, default=2025, help="Input season (default: 2025)")
    parser.add_argument("--top",          type=int, default=20,   help="Players to show per position")
    args = parser.parse_args()

    if not args.predict_only:
        train_all()

    predict_all(season=args.season, top_n_display=args.top)
