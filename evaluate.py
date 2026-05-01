"""
evaluate.py — Validate model predictions against actual 2025 results.

Each position gets a 3-panel figure:
  1. Predicted vs. Actual scatter — color-coded by outcome category:
       GREEN  = model nailed an elite player
       RED    = model missed badly, no injury excuse
       ORANGE = surprise breakout — model underrated, player outperformed
       GRAY   = injury-shortened season (faded, not the focus)
       BLUE   = everyone else
  2. Learning Curve  — does more training data improve accuracy?
  3. Permutation Importance — which features actually move the needle?

Usage:
    python evaluate.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import PredictionErrorDisplay, mean_absolute_error, r2_score
from sklearn.model_selection import LearningCurveDisplay, KFold

from db import get_seasons
from features import (
    POSITIONS, _load_full, build_training_data, engineer,
    add_durability_features, INJURY_GAMES_THRESHOLD,
)

MODEL_DIR  = Path("models")
OUTPUT_DIR = Path("evaluation")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Thresholds for annotation categories ---
# A prediction error larger than this (and no injury) = model whiff
BIG_MISS_THRESHOLD   = 45   # PPR points
# A prediction error smaller than this on an elite player = model hit
GOOD_HIT_THRESHOLD   = 25   # PPR points
# "Elite" = predicted PPR is above this percentile for the position
ELITE_PERCENTILE     = 60
# Breakout = actual was this much MORE than predicted, player wasn't injured
BREAKOUT_THRESHOLD   = 50   # PPR points


def load_model(position: str):
    path = MODEL_DIR / f"{position.lower()}_model.pkl"
    if not path.exists():
        return None, None
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_cols"]


def get_holdout_data(position: str, train_season: int = 2024, eval_season: int = 2025):
    """
    Build a true out-of-sample test set.
    Returns merged DataFrame with 2024 features + 2025 actuals + 2025 games played.
    """
    df = _load_full(position)

    feat  = df[df["season"] == train_season].copy()
    actuals = df[df["season"] == eval_season][["player_id", "ppr_points", "games"]].copy()
    actuals = actuals.rename(columns={"ppr_points": "actual_ppr", "games": "actual_games"})

    merged = feat.merge(actuals, on="player_id", how="inner")
    return merged


def _categorize(y_pred, y_true, actual_games, feature_cols, X):
    """
    Assign each player a display category based on prediction outcome.
    Returns a list of category strings, one per player.
    """
    n = len(y_pred)
    error = y_pred - y_true
    abs_error = np.abs(error)
    injured = actual_games < INJURY_GAMES_THRESHOLD
    elite_cutoff = np.percentile(y_pred, ELITE_PERCENTILE)

    cats = []
    for i in range(n):
        if injured.iloc[i]:
            cats.append("injured")
        elif abs_error[i] <= GOOD_HIT_THRESHOLD and y_pred[i] >= elite_cutoff:
            cats.append("hit")                    # model nailed an expected good player
        elif error[i] < -BREAKOUT_THRESHOLD:
            cats.append("breakout")               # player blew past expectations
        elif abs_error[i] >= BIG_MISS_THRESHOLD:
            cats.append("miss")                   # model was confidently wrong
        else:
            cats.append("normal")
    return cats


def evaluate_position(position: str) -> dict | None:
    model, feature_cols = load_model(position)
    if model is None:
        print(f"[{position}] No model found — run model.py first.")
        return None

    holdout = get_holdout_data(position)
    if len(holdout) < 5:
        print(f"[{position}] Not enough holdout data ({len(holdout)} players).")
        return None

    X_hold = holdout.reindex(columns=feature_cols).fillna(0)
    y_true = holdout["actual_ppr"].values
    y_pred = model.predict(X_hold)

    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"[{position}]  n={len(holdout):3d}  MAE={mae:.1f}  R²={r2:.3f}")

    X_train, y_train, _ = build_training_data(position)

    return {
        "position":     position,
        "model":        model,
        "feature_cols": feature_cols,
        "holdout":      holdout,
        "X_hold":       X_hold,
        "y_true":       y_true,
        "y_pred":       y_pred,
        "X_train":      X_train,
        "y_train":      y_train,
        "mae":          mae,
        "r2":           r2,
    }


def plot_position(res: dict) -> None:
    pos     = res["position"]
    holdout = res["holdout"]
    y_true  = res["y_true"]
    y_pred  = res["y_pred"]
    names   = holdout["player_name"].values
    actual_games = holdout["actual_games"]

    cats = _categorize(y_pred, y_true, actual_games, res["feature_cols"], res["X_hold"])

    STYLE = {
        "hit":      dict(color="#2ca02c", marker="*", s=120, zorder=5, label="Model nailed it (elite)"),
        "miss":     dict(color="#d62728", marker="X", s=100, zorder=5, label="Big miss — no injury"),
        "breakout": dict(color="#ff7f0e", marker="^", s=100, zorder=5, label="Surprise breakout"),
        "injured":  dict(color="#aaaaaa", marker="o", s=25,  zorder=2, alpha=0.4, label="Injury-shortened"),
        "normal":   dict(color="#1f77b4", marker="o", s=40,  zorder=3, alpha=0.6, label="Normal"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        f"{pos}  —  2024→2025 Holdout Evaluation  |  MAE={res['mae']:.1f} PPR pts  |  R²={res['r2']:.3f}",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: Predicted vs Actual, color-coded ────────────────────────
    ax = axes[0]
    cats_arr = np.array(cats)

    for cat, style in STYLE.items():
        mask = cats_arr == cat
        if not mask.any():
            continue
        kw = {k: v for k, v in style.items() if k != "label"}
        ax.scatter(y_pred[mask], y_true[mask], **kw)

    # Perfect-prediction diagonal
    lo = min(y_pred.min(), y_true.min()) - 10
    hi = max(y_pred.max(), y_true.max()) + 10
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted 2025 PPR")
    ax.set_ylabel("Actual 2025 PPR")
    ax.set_title("Predicted vs. Actual 2025 PPR")

    # Annotate hits and misses — NOT just whoever was most wrong
    abs_err = np.abs(y_pred - y_true)

    # Top hits: best predictions on elite players
    hit_idx = np.where(cats_arr == "hit")[0]
    if len(hit_idx):
        top_hits = hit_idx[np.argsort(abs_err[hit_idx])[:5]]
        for i in top_hits:
            ax.annotate(
                names[i],
                xy=(y_pred[i], y_true[i]),
                color="#2ca02c", fontsize=7.5, fontweight="bold",
                xytext=(5, 3), textcoords="offset points",
            )

    # Top misses: biggest errors that weren't injuries
    miss_idx = np.where(cats_arr == "miss")[0]
    if len(miss_idx):
        top_misses = miss_idx[np.argsort(-abs_err[miss_idx])[:5]]
        for i in top_misses:
            ax.annotate(
                f"{names[i]}\n(pred {y_pred[i]:.0f} / got {y_true[i]:.0f})",
                xy=(y_pred[i], y_true[i]),
                color="#d62728", fontsize=7, fontweight="bold",
                xytext=(5, -12), textcoords="offset points",
            )

    # Breakouts
    brk_idx = np.where(cats_arr == "breakout")[0]
    if len(brk_idx):
        top_brk = brk_idx[np.argsort(y_true[brk_idx] - y_pred[brk_idx])[-4:]]
        for i in top_brk:
            ax.annotate(
                names[i],
                xy=(y_pred[i], y_true[i]),
                color="#ff7f0e", fontsize=7.5, fontweight="bold",
                xytext=(5, 3), textcoords="offset points",
            )

    # Legend
    legend_handles = [
        mpatches.Patch(color=STYLE[c]["color"], label=STYLE[c]["label"])
        for c in ["hit", "miss", "breakout", "normal", "injured"]
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, loc="upper left")

    # ── Panel 2: Learning Curve ──────────────────────────────────────────
    ax = axes[1]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    LearningCurveDisplay.from_estimator(
        res["model"], res["X_train"], res["y_train"],
        cv=cv, scoring="neg_mean_absolute_error",
        n_jobs=-1, ax=ax,
        std_display_style="fill_between",
        score_name="MAE", score_type="both",
    )
    ax.set_title("Learning Curve  (does more data help?)")
    ax.set_xlabel("Training samples")

    # ── Panel 3: Permutation Importance ─────────────────────────────────
    ax = axes[2]
    perm = permutation_importance(
        res["model"], res["X_hold"], y_true,
        n_repeats=20, random_state=42, n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )
    order = np.argsort(perm.importances_mean)
    cols  = res["feature_cols"]

    # Color durability features differently so they stand out
    dur_names = {"games_pct", "prev_games_pct", "career_durability", "injury_last2"}
    bar_colors = [
        "#e377c2" if cols[i] in dur_names else "steelblue"
        for i in order
    ]
    ax.barh(
        [cols[i] for i in order],
        perm.importances_mean[order],
        xerr=perm.importances_std[order],
        color=bar_colors, ecolor="gray", capsize=3,
    )
    ax.set_title("Permutation Importance  (holdout set)")
    ax.set_xlabel("Mean MAE increase when feature shuffled")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    dur_patch   = mpatches.Patch(color="#e377c2", label="Durability/injury feature")
    other_patch = mpatches.Patch(color="steelblue", label="Performance feature")
    ax.legend(handles=[dur_patch, other_patch], fontsize=8, loc="lower right")

    plt.tight_layout()
    out = OUTPUT_DIR / f"{pos.lower()}_evaluation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def print_category_breakdown(res: dict) -> None:
    """Print a text summary of who fell into each category."""
    holdout = res["holdout"]
    y_true  = res["y_true"]
    y_pred  = res["y_pred"]
    cats    = _categorize(y_pred, y_true, holdout["actual_games"],
                          res["feature_cols"], res["X_hold"])
    abs_err = np.abs(y_pred - y_true)
    names   = holdout["player_name"].values

    for cat, label in [
        ("hit",      "Model hits on elite players"),
        ("miss",     "Big misses — no injury excuse"),
        ("breakout", "Surprise breakouts"),
    ]:
        idx = [i for i, c in enumerate(cats) if c == cat]
        if not idx:
            continue
        idx_sorted = sorted(idx, key=lambda i: -abs_err[i] if cat != "hit" else abs_err[i])
        print(f"\n  {label}:")
        for i in idx_sorted[:6]:
            print(f"    {names[i]:<22}  pred={y_pred[i]:6.1f}  actual={y_true[i]:6.1f}  "
                  f"err={y_pred[i]-y_true[i]:+.1f}  games={holdout['actual_games'].iloc[i]:.0f}")


def run_all() -> None:
    print("=" * 55)
    print("  NFL PPR Predictor — Model Evaluation (2024 → 2025)")
    print("=" * 55)

    summary_rows = []
    for pos in POSITIONS:
        res = evaluate_position(pos)
        if res is None:
            continue
        print_category_breakdown(res)
        plot_position(res)
        summary_rows.append({"Pos": pos, "MAE": round(res["mae"], 1), "R²": round(res["r2"], 3)})

    print(f"\n{'─'*30}")
    print("Summary")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nCharts saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_all()
