# NFL Player Performance Predictor

A machine learning pipeline that predicts NFL player fantasy PPR scoring for the upcoming season and estimates the probability of finishing as a top-6, top-12, or top-24 player at their position.

Built with Python, scikit-learn, pandas, and SQLite.

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

WR and TE beat the naive baseline. QB is the hardest to predict(e.g., Lamar Jackson going from a 471-point 2024 to 215 in 2025 due to games missed.) Offenses also change drastically.

---

## Key Engineering Decisions

**Why Ridge over GradientBoosting for most positions?**
With ~200–450 training samples per position, Ridge's linear assumptions generalize better than a complex tree ensemble. The model automatically cross-validates both and picks the winner per position.

**Two-season lag features**
Rather than only using last year's stats, the model looks at the prior two seasons. This is critical for injury recovery or a season with a bad quarterback. A player who scored 400 PPR in year N-2 but only 120 in year N-1 (due to injury) should not be treated the same as someone consistently averaging 120. This helped players like Chris Olave get a boost, who had a great 2023 and a bad 2024.

**SQL + pandas together**
Raw data lives in a SQLite database. All data loading goes through parameterized SQL queries (`db.py`), with pandas handling transformation and feature engineering downstream.

## Tech Stack

- **Data** — [nflverse-data](https://github.com/nflverse/nflverse-data) (parquet via PyArrow)
- **Storage** — SQLite (`sqlite3`)
- **Processing** — pandas, NumPy
- **ML** — scikit-learn (`GradientBoostingRegressor`, `Ridge`, `CalibratedClassifierCV`, `Pipeline`, `StandardScaler`)
- **Evaluation** — `PredictionErrorDisplay`, `LearningCurveDisplay`, `permutation_importance`
- **Serialization** — joblib

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
└── season_totals_2019_2025.csv   # Pre-pulled data
```

---

## How to Run

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Full pipeline** (first time or to refresh with latest data)
```bash
python dataExtraction.py   # pulls data from nflverse
python db.py               # builds SQLite database
python model.py            # trains regression models and saves to models
python classify.py         # trains finish-probability classifiers
python evaluate.py         # generates evaluation charts that go to evaluation graphs
python predict.py          # prints 2026 PPR projections
```

---

## Sample Output

**2026 PPR Projections — WR (top 10)**
```
    Player       Team   Age  GP  2024 PPR  Pred 2025 PPR
1   J.Chase      CIN   24.50  17   403.0         306.7
2   A.Sr. Brown  DET   24.86  17   316.18        278.5 
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

NFL year-to-year prediction is difficult, with how many factors go into a players success year after year. The biggest sources of error are:
- **Breakouts** — players who step into a new role the model hasn't seen.
- **Team/role changes** — a WR, a WR's QB, changing teams completely changes their opportunity. Also doesn't factor in Offensive-Line health/changes enough.
- **SMALL TRAINING SET** - The biggest issue with using ML for Fantasy Football, there being not enough data. We can't use every season of the NFL since NFL in the 80s was so much different than it was now.

Pure historical stats probably aren't enough to beat a sharp human analyst or a Vegas-calibrated model. Where ML would shine in the fantasy football space is if I combined it with those external signals, such as PFF grades and other things like that.
