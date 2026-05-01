"""
model.py — Train per-position PPR predictors and compare GradientBoosting vs Ridge.

For each position, both models are cross-validated and the winner is saved.
Ridge uses a StandardScaler pipeline since it's sensitive to feature scale.

Usage:
    python model.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import POSITIONS, build_training_data

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def _cv_mae(model, X, y, cv) -> float:
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    return -scores.mean()


def train_position(position: str) -> dict:
    X, y, feature_cols = build_training_data(position)

    if len(X) < 20:
        print(f"{position}: too few samples ({len(X)}), skipping.")
        return {}

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # ── Gradient Boosting ──────────────────────────────────────────────
    gb = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42,
    )
    gb_cv_mae = _cv_mae(gb, X, y, cv)

    # ── Ridge Regression (needs scaling — wrapped in a Pipeline) ───────
    # Alpha controls regularization strength; cross-validate a few values
    best_ridge_mae = np.inf
    best_alpha     = 1.0
    for alpha in [0.1, 1.0, 10.0, 50.0, 100.0, 500.0]:
        ridge_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=alpha)),
        ])
        mae = _cv_mae(ridge_pipe, X, y, cv)
        if mae < best_ridge_mae:
            best_ridge_mae = mae
            best_alpha     = alpha

    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=best_alpha)),
    ])
    ridge_cv_mae = best_ridge_mae

    # ── Pick winner ────────────────────────────────────────────────────
    winner_name = "GradientBoosting" if gb_cv_mae <= ridge_cv_mae else "Ridge"
    winner      = gb                  if gb_cv_mae <= ridge_cv_mae else ridge
    winner_mae  = min(gb_cv_mae, ridge_cv_mae)

    winner.fit(X, y)
    train_mae = mean_absolute_error(y, winner.predict(X))
    train_r2  = r2_score(y, winner.predict(X))

    model_path = MODEL_DIR / f"{position.lower()}_model.pkl"
    joblib.dump({"model": winner, "feature_cols": feature_cols}, model_path)

    margin = abs(gb_cv_mae - ridge_cv_mae)
    print(
        f"[{position}]  n={len(X)}\n"
        f"  GradientBoosting  CV MAE={gb_cv_mae:.1f}\n"
        f"  Ridge (α={best_alpha:<6})  CV MAE={ridge_cv_mae:.1f}\n"
        f"  → Winner: {winner_name}  (by {margin:.1f} pts)  "
        f"Train MAE={train_mae:.1f}  Train R²={train_r2:.3f}\n"
    )
    return {
        "position":     position,
        "winner":       winner_name,
        "model":        winner,
        "feature_cols": feature_cols,
        "cv_mae":       winner_mae,
        "gb_cv_mae":    gb_cv_mae,
        "ridge_cv_mae": ridge_cv_mae,
        "n":            len(X),
    }


def plot_results(results: list[dict]) -> None:
    valid = [r for r in results if r]
    if not valid:
        return

    # ── Feature importance (GB only — Ridge uses coefficients instead) ──
    gb_results = [r for r in valid if r["winner"] == "GradientBoosting"]
    if gb_results:
        n   = len(gb_results)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 7))
        if n == 1:
            axes = [axes]
        for ax, res in zip(axes, gb_results):
            imp   = res["model"].feature_importances_
            cols  = res["feature_cols"]
            order = np.argsort(imp)

            history_features = {
                "lag1_ppr", "lag2_ppr", "ppr_trend", "peak_ppr_2yr",
                "lag1_ppr_per_game", "lag1_top6", "lag1_top12", "team_changed",
            }
            colors = ["#e377c2" if cols[i] in history_features else "steelblue"
                      for i in order]

            ax.barh([cols[i] for i in order], imp[order], color=colors)
            ax.set_title(
                f"{res['position']} (GradientBoosting won)\n"
                f"CV MAE={res['cv_mae']:.1f}  n={res['n']}",
                fontsize=11,
            )
            ax.set_xlabel("Feature importance")

        pink  = plt.Rectangle((0,0),1,1, fc="#e377c2")
        blue  = plt.Rectangle((0,0),1,1, fc="steelblue")
        axes[0].legend([pink, blue], ["History/durability feature", "Current-season feature"],
                       fontsize=8)
        plt.suptitle("NFL PPR Predictor — Feature Importance (GradientBoosting positions)",
                     fontsize=13, y=1.01)
        plt.tight_layout()
        out = Path("feature_importance.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

    # ── Ridge coefficients for Ridge-winning positions ──────────────────
    ridge_results = [r for r in valid if r["winner"] == "Ridge"]
    if ridge_results:
        n   = len(ridge_results)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 7))
        if n == 1:
            axes = [axes]
        for ax, res in zip(axes, ridge_results):
            pipe  = res["model"]
            coefs = pipe.named_steps["ridge"].coef_
            cols  = res["feature_cols"]
            order = np.argsort(np.abs(coefs))
            colors = ["#d62728" if coefs[i] < 0 else "steelblue" for i in order]

            ax.barh([cols[i] for i in order], coefs[order], color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(
                f"{res['position']} (Ridge won, α={pipe.named_steps['ridge'].alpha})\n"
                f"CV MAE={res['cv_mae']:.1f}  n={res['n']}",
                fontsize=11,
            )
            ax.set_xlabel("Coefficient (positive = higher PPR next year)")

        plt.suptitle("NFL PPR Predictor — Ridge Coefficients", fontsize=13, y=1.01)
        plt.tight_layout()
        out = Path("ridge_coefficients.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")


def train_all() -> None:
    print("Training per-position models (GradientBoosting vs Ridge)...\n")
    results = [train_position(pos) for pos in POSITIONS]
    plot_results(results)

    print("\nModel selection summary:")
    print(f"  {'Pos':<4} {'Winner':<18} {'GB MAE':>8} {'Ridge MAE':>10}")
    print(f"  {'-'*44}")
    for r in results:
        if r:
            print(f"  {r['position']:<4} {r['winner']:<18} {r['gb_cv_mae']:>8.1f} {r['ridge_cv_mae']:>10.1f}")


if __name__ == "__main__":
    train_all()
