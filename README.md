# FIFA World Cup 2026 Prediction Project

## Model Comparison Results

Running `uv run model_comparison_2.py` evaluates multiple classifiers on historical match data with knockout-stage features (stage weight, rest days, penalty shootout flag, 4-class target encoding).

### Current Validation Accuracies (80/20 split, stratified, 101 matches with scores)

| Model | Full Validation | Warm-up Excluded* |
|-------|----------------|-------------------|
| TabICL | **80.95%** | **84.62%** |
| Random Forest | 71.43% | 69.23% |
| Logistic Regression (scaled) | 66.67% | 69.23% |
| Gradient Boosting | 57.14% | 46.15% |
| Kumo-Tabular (small) | 4.76%† | 0.00%† |
| XGBoost | not installed | — |
| TypeSafe | API key required | — |

*Warm-up excluded: drops each team's first non-tactical match and all tactical matches (park-the-bus draws).

†Kumo-Tabular shows **extreme variance** (0–76% across runs) due to its in-context learning with random ECOC codebooks. On this small dataset (80 train / 21 val), the foundation model's stochastic codebook sampling dominates signal. Not recommended for production use here.

### Data Characteristics Driving These Results

1. **Very small sample** — Only ~100 labeled matches total; ~80 for training after warm-up exclusion.
2. **Class imbalance** — 4-class target: Draw (20), Home win (49), Away win (28), Penalty win (4).
3. **Feature set** — 17 numerical features (base/form attack/defense, appearances, injuries, rank diff, knockout flag, stage weight, rest days, penalty flag).
4. **Tactical filtering** — 10 matches marked `is_tactical=1` excluded from form updates and warm-up calculation.
5. **No XGBoost** — OpenMP runtime missing in environment; XGBoost unavailable.

### Recommendations

- **TabICL** is the strongest performer for this data regime (tabular foundation model with ICL, deterministic).
- **Random Forest** is a solid, stable baseline.
- **Logistic Regression** with scaling performs reasonably and is interpretable.
- **Kumo-Tabular** needs more data or fine-tuning; current small-model ICL is too high-variance.
- Consider **collecting more historical matches** (friendlies, qualifiers) to expand training set.
- For production: ensemble TabICL + Random Forest + Logistic Regression.

## Usage

```bash
# Install dependencies
uv sync

# Run model comparison (knockout-stage features)
uv run model_comparison_2.py

# Run main prediction pipeline
uv run predict.py
```

## Data Sources

- `data/teams.csv` — World rankings, appearances, base attack/defense
- `data/matches.csv` — Tournament matches (with `is_tactical`, `match_stage`, `is_penalty_shootout`)
- `data/friendlies.csv` — Friendly matches (lower weight)
- `data/player_status.csv` — Key player injuries/suspensions (`is_key` flag)