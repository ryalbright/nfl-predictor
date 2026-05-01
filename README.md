# NFL Player Performance Predictor

A machine learning pipeline that predicts NFL player fantasy PPR scoring for the upcoming season and estimates the probability of finishing as a top-6, top-12, or top-24 player at their position.

Built with Python, scikit-learn, pandas, and SQLite.

---

## What It Does

- **Pulls live NFL data** from the [nflverse](https://github.com/nflverse/nflverse-data) dataset (2019–2025 regular season)
- **Stores it in a SQLite database** with a structured schema, queryable via SQL
- **Engineers features** across two seasons of history per player: per-game rates, efficiency metrics, injury/durability signals, positional rank history, and team change flags
- **Trains per-position regression models** (Ridge or GradientBoosting — whichever cross-validates better) to predict next season's PPR total
- **Trains per-position classifiers** to output finish probabilities: top-6, top-12, and top-24
- **Evaluates honestly** on a true out-of-sample holdout (2024 stats → 2025 actuals)

---

## Results (2024 → 2025 Holdout)

| Position | Model | MAE | R² | Naive baseline R² |
|----------|-------|-----|-----|-------------------|
| QB | Ridge | 67.7 | 0.491 | 0.384 |
| RB | GradientBoosting | 51.6 | 0.489 | 0.508 |
| WR | Ridge | 38.8 | 0.588 | 0.405 |
| TE | Ridge | 27.5 | 0.638 | 0.529 |

*Naive baseline = predict every player scores exactly the same as last year.*

WR and TE beat the naive baseline. QB is the hardest to predict — one injury to an elite QB (e.g., Lamar Jackson going from a 471-point 2024 to 215 in 2025 due to games missed) tanks the whole metric.

---

## Key Engineering Decisions

**Why Ridge over GradientBoosting for most positions?**
With ~200–450 training samples per position, Ridge's linear assumptions generalize better than a complex tree ensemble. The model automatically cross-validates both and picks the winner per position.

**Two-season lag features**
Rather than only using last year's stats, the model looks at the prior two seasons. This is critical for injury recovery — a player who scored 400 PPR in year N-2 but only 120 in year N-1 (due to injury) should not be treated the same as someone consistently averaging 120.

**Honest evaluation**
An early version of this project had a data leakage bug where the 2024→2025 transition was included in both training and evaluation, making holdout metrics look artificially good (TE R² was 0.932 — essentially testing on training data). Catching and fixing this brought the TE R² to an honest 0.638.

**SQL + pandas together**
Raw data lives in a SQLite database. All data loading goes through parameterized SQL queries (`db.py`), with pandas handling transformation and feature engineering downstream.

---

## Tech Stack

- **Data** — [nflverse-data](https://github.com/nflverse/nflverse-data) (parquet via PyArrow)
- **Storage** — SQLite (`sqlite3`)
- **Processing** — pandas, NumPy
- **ML** — scikit-learn (`GradientBoostingRegressor`, `Ridge`, `CalibratedClassifierCV`, `Pipeline`, `StandardScaler`)
- **Evaluation** — `PredictionErrorDisplay`, `LearningCurveDisplay`, `permutation_importance`
- **Serialization** — joblib

---

## Project Structure

```
├── dataExtraction.py   # Pull 2019–2025 regular season data from nflverse
├── db.py               # Load CSV → SQLite; SQL query helpers
├── features.py         # Feature engineering (rates, lags, durability, rank history)
├── model.py            # Train regression models (GB vs Ridge comparison)
├── classify.py         # Train finish-probability classifiers (top 6/12/24)
├── evaluate.py         # Holdout evaluation with categorized scatter plots
├── predict.py          # Generate 2026 PPR projections
├── requirements.txt
└── season_totals_2019_2025.csv   # Pre-pulled data (skip dataExtraction.py if using this)
```

---

## How to Run

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Full pipeline** (first time or to refresh with latest data)
```bash
python dataExtraction.py   # ~1 min — pulls fresh data from nflverse
python db.py               # builds SQLite database
python model.py            # trains regression models, saves to models/
python classify.py         # trains finish-probability classifiers
python evaluate.py         # generates evaluation charts → evaluation/
python predict.py          # prints 2026 PPR projections
```

**Skip data pull** (using the included CSV)
```bash
python db.py
python model.py
python classify.py
python evaluate.py
python predict.py
```

---

## Sample Output

**2026 PPR Projections — TE (top 10)**
```
    Player       Team   Age  GP  2025 PPR  Pred 2026 PPR
1   T.McBride    ARI   25.8  17    315.9         ...
2   C.Loveland   CHI   21.4  18    198.4         ...
...
```

**2026 Finish Probabilities — RB**
```
    Player       Team   Top-6 %  Top-12 %  Top-24 %
1   J.Gibbs      DET     43.8      54.8      75.2
2   D.Achane     MIA     48.6      55.2      84.8
...
```

---

## Limitations

NFL year-to-year prediction is genuinely hard. The biggest sources of error are:
- **Breakouts** — players who step into a new role the model hasn't seen (e.g., Amon-Ra St. Brown 2025)
- **Team/role changes** — a WR changing teams completely changes their opportunity
- **Small training set** — only 6 seasons of data; more historical seasons would meaningfully improve accuracy

These aren't bugs — they reflect real uncertainty in NFL outcomes that even professional analysts can't fully predict.
