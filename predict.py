"""
predict.py — Generate 2025 PPR predictions using trained models.

Usage:
    python predict.py              # top 15 per position
    python predict.py --top 30    # top 30 per position
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import POSITIONS, build_predict_data

MODEL_DIR = Path("models")


def predict_position(position: str, season: int = 2024, top_n: int = 15) -> pd.DataFrame:
    model_path = MODEL_DIR / f"{position.lower()}_model.pkl"
    if not model_path.exists():
        print(f"  [{position}] No model found — run model.py first.")
        return pd.DataFrame()

    bundle      = joblib.load(model_path)
    model       = bundle["model"]
    feature_cols = bundle["feature_cols"]

    df = build_predict_data(position, season=season, feature_cols=feature_cols)
    if df.empty:
        print(f"  [{position}] No {season} data found.")
        return pd.DataFrame()

    X = df[feature_cols].fillna(0)
    df = df.copy()
    df["predicted_ppr_2025"] = model.predict(X).round(1)

    out = (
        df[["player_name", "team", "age_sept1", "games", "ppr_points", "predicted_ppr_2025"]]
        .rename(columns={
            "player_name":         "Player",
            "team":                "Team",
            "age_sept1":           "Age",
            "games":               "GP",
            "ppr_points":          "2024 PPR",
            "predicted_ppr_2025":  "Pred 2025 PPR",
        })
        .sort_values("Pred 2025 PPR", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    out.index += 1
    return out


def run_all(season: int = 2024, top_n: int = 15) -> None:
    for pos in POSITIONS:
        print(f"\n{'─' * 55}")
        print(f"  {pos}  —  Predicted 2025 PPR Points  (based on {season})")
        print(f"{'─' * 55}")
        result = predict_position(pos, season=season, top_n=top_n)
        if not result.empty:
            print(result.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict 2025 NFL PPR scores.")
    parser.add_argument("--top",    type=int, default=15, help="Players to show per position")
    parser.add_argument("--season", type=int, default=2024, help="Input season to predict from")
    args = parser.parse_args()

    run_all(season=args.season, top_n=args.top)
