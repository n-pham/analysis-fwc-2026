"""Model comparison script

- Implements a lightweight logistic‑regression baseline.
- Trains a shallow XGBoost model with log‑scaled opponent weighting (already used
  in ``predict.py``) and early stopping.
- Uses the same feature engineering pipeline as ``predict.py`` (base attack,
  base defense, form, pedigree, injuries, rank diff, home‑host bonus).
- Prints overall accuracy and the warm‑up‑excluded accuracy (which also drops
  tactical matches) for both models.
"""

from __future__ import annotations

import math
import numpy as np
import polars as pl
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# XGBoost is optional – if unavailable we fall back to logistic regression only.
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:  # pragma: no cover
    XGB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Helper utilities (mirroring parts of ``predict.py``)
# ---------------------------------------------------------------------------

def load_data():
    teams = pl.read_csv("data/teams.csv")
    matches = pl.read_csv("data/matches.csv")
    # Cast score columns to Int64 (blank cells become null)
    matches = matches.with_columns([
        pl.col("score_home").cast(pl.Int64, strict=False),
        pl.col("score_away").cast(pl.Int64, strict=False),
    ])
    try:
        friendlies = pl.read_csv("data/friendlies.csv")
    except Exception:
        friendlies = None
    try:
        player_status = pl.read_csv("data/player_status.csv")
    except Exception:
        player_status = None
    return teams, matches, friendlies, player_status

# Re‑use the form‑score calculation (it already skips tactical matches).
def get_form_scores(friendlies, matches, teams_df):
    # Very similar to the function in ``predict.py`` – copied here for independence.
    teams_list = teams_df["team"].to_list()
    rank_map = {row["team"]: row["world_ranking"] for row in teams_df.to_dicts()}
    form = {team: {"attack": 0.0, "defense": 0.0} for team in teams_list}

    # Friendlies – low weight (1.0)
    if friendlies is not None:
        for r in friendlies.to_dicts():
            h, a, sh, sa = r["team_home"], r["team_away"], r["score_home"], r["score_away"]
            if sh is None or sa is None:
                continue
            if h in form:
                form[h]["attack"] += sh * 1.0
                if sa == 0:
                    form[h]["defense"] += 2.0
            if a in form:
                form[a]["attack"] += sa * 1.0
                if sh == 0:
                    form[a]["defense"] += 2.0

    # Tournament matches – higher weight + opponent rank factor
    for r in matches.to_dicts():
        h, a, sh, sa = (
            r["team_home"],
            r["team_away"],
            r["score_home"],
            r["score_away"],
        )
        if r.get("is_tactical", 0) == 1:
            continue
        if sh is None or sa is None:
            continue
        h_rank = rank_map.get(h, 50)
        a_rank = rank_map.get(a, 50)
        # Log‑scaled rank weight (mirrors predict.py change)
        h_weight = max(0.1, math.log1p(101 - a_rank) / math.log1p(100))
        a_weight = max(0.1, math.log1p(101 - h_rank) / math.log1p(100))
        if h in form:
            form[h]["attack"] += sh * 5.0 * h_weight
            if sa == 0:
                form[h]["defense"] += 10.0 * h_weight
            penalty = sa * 3.0 * (1.0 / h_weight)
            if sh - sa >= 3:
                penalty *= 0.1
            form[h]["defense"] -= penalty
        if a in form:
            form[a]["attack"] += sa * 5.0 * a_weight
            if sh == 0:
                form[a]["defense"] += 10.0 * a_weight
            penalty = sh * 3.0 * (1.0 / a_weight)
            if sa - sh >= 3:
                penalty *= 0.1
            form[a]["defense"] -= penalty
    return form

def calculate_injuries(matches, player_status):
    # Same logic as ``predict.py`` – returns a DataFrame with injury counts.
    inj_h, inj_a = [], []
    for r in matches.to_dicts():
        m_id = r["match_id"]
        h, a = r["team_home"], r["team_away"]
        inj_home = inj_away = 0
        if player_status is not None:
            for side, team in (("home", h), ("away", a)):
                filt = player_status.filter((pl.col("team") == team) & (pl.col("is_key") == True))
                for p in filt.to_dicts():
                    unavail = [x.strip() for x in str(p["unavailable_match_ids"]).split(",")]
                    if str(m_id) in unavail:
                        if side == "home":
                            inj_home += 1
                        else:
                            inj_away += 1
        inj_h.append(inj_home)
        inj_a.append(inj_away)
    return matches.with_columns([
        pl.Series(name="injuries_home", values=inj_h),
        pl.Series(name="injuries_away", values=inj_a),
    ])

# ---------------------------------------------------------------------------
# Feature engineering for ML models
# ---------------------------------------------------------------------------

def build_feature_df(matches_df: pl.DataFrame, teams_df: pl.DataFrame, form_scores: dict) -> pl.DataFrame:
    # Attach form scores to teams
    form_df = pl.DataFrame({
        "team": list(form_scores.keys()),
        "form_attack": [v["attack"] for v in form_scores.values()],
        "form_defense": [v["defense"] for v in form_scores.values()],
    })
    teams = teams_df.join(form_df, on="team", how="left")

    # Replace team names (slot mapping – empty for now)
    mapping = {}
    matches = matches_df.with_columns([
        pl.col("team_home").replace(mapping).alias("team_home"),
        pl.col("team_away").replace(mapping).alias("team_away"),
    ])

    # Join home and away team features
    matches = matches.join(
        teams.select([
            pl.col("team").alias("team_home"),
            pl.col("world_ranking").alias("rank_home"),
            pl.col("appearances").alias("apps_home"),
            pl.col("base_attack").alias("ba_home"),
            pl.col("base_defense").alias("bd_home"),
            pl.col("form_attack").alias("fa_home"),
            pl.col("form_defense").alias("fd_home"),
        ]),
        on="team_home",
        how="left",
    )
    matches = matches.join(
        teams.select([
            pl.col("team").alias("team_away"),
            pl.col("world_ranking").alias("rank_away"),
            pl.col("appearances").alias("apps_away"),
            pl.col("base_attack").alias("ba_away"),
            pl.col("base_defense").alias("bd_away"),
            pl.col("form_attack").alias("fa_away"),
            pl.col("form_defense").alias("fd_away"),
        ]),
        on="team_away",
        how="left",
    )

    # Create numeric helper columns
    matches = matches.with_columns([
        # Rank difference (positive => home is stronger)
        (pl.col("rank_home") - pl.col("rank_away")).alias("rank_diff"),
        # Absolute rank difference (magnitude only)
        (pl.col("rank_home") - pl.col("rank_away")).abs().alias("abs_rank_diff"),
        # Host bonus expressed as the actual 10 points used in predict_logic
        pl.when(pl.col("team_home").is_in(["Mexico", "Canada", "USA"]))
        .then(10)
        .otherwise(0)
        .alias("host_bonus"),
    ])
    return matches

# ---------------------------------------------------------------------------
# Target encoding – three classes: 0=Draw, 1=Home win, 2=Away win
# ---------------------------------------------------------------------------

def encode_target(df: pl.DataFrame) -> pl.Series:
    """Encode the match outcome as a numeric label.

    Returns a Polars Series with values:
    * 0 – Draw
    * 1 – Home win
    * 2 – Away win
    Rows with missing scores are encoded as ``None`` (they will be filtered out
    later)."""
    # Convert to pandas for a simple row‑wise apply (the dataset is tiny).
    import pandas as pd

    def map_outcome(row: pd.Series):
        sh, sa = row["score_home"], row["score_away"]
        if sh == "" or sa == "" or pd.isna(sh) or pd.isna(sa):
            return None
        sh, sa = int(sh), int(sa)
        if sh > sa:
            return 1
        if sa > sh:
            return 2
        return 0

    pd_series = df.to_pandas().apply(map_outcome, axis=1)
    return pl.Series(pd_series)


# ---------------------------------------------------------------------------
# Main comparison routine
# ---------------------------------------------------------------------------

def main():
    teams, matches, friendlies, player_status = load_data()
    form_scores = get_form_scores(friendlies, matches, teams)
    matches = calculate_injuries(matches, player_status)
    feature_df = build_feature_df(matches, teams, form_scores)

    # Keep only rows with actual scores (to evaluate predictive power)
    actual_mask = (pl.col("score_home").is_not_null()) & (pl.col("score_away").is_not_null())
    eval_df = feature_df.filter(actual_mask)

    # Identify first non-tactical match for each team (warm-up period)
    non_tactical_df = eval_df.filter(pl.col("is_tactical") != 1)
    home = non_tactical_df.select([
        pl.col("match_id"),
        pl.col("date"),
        pl.col("team_home").alias("team"),
    ])
    away = non_tactical_df.select([
        pl.col("match_id"),
        pl.col("date"),
        pl.col("team_away").alias("team"),
    ])
    long = pl.concat([home, away])
    long = long.sort(["team", "date", "match_id"], descending=False)
    first_per_team = long.group_by("team").agg(pl.col("match_id").first().alias("first_match_id"))
    first_match_ids = set(first_per_team["first_match_id"].to_list())

    # Encode target
    y = encode_target(eval_df).to_numpy()
    # Drop rows where target is None (should not happen after mask)
    valid_idx = ~np.isnan(y)
    
    # Store match metadata to align with split
    match_ids = eval_df["match_id"].to_numpy()[valid_idx]
    is_tactical = eval_df["is_tactical"].to_numpy()[valid_idx]

    X = eval_df.select([
        "ba_home", "ba_away", "bd_home", "bd_away",
        "fa_home", "fa_away", "fd_home", "fd_away",
        "apps_home", "apps_away",
        "injuries_home", "injuries_away",
        "rank_diff", "host_bonus",
    ]).to_numpy()[valid_idx]
    y = y[valid_idx].astype(int)

    # Simple train/val split (80/20)
    indices = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    val_match_ids = match_ids[val_idx]
    val_is_tactical = is_tactical[val_idx]

    # Validation evaluation mask (excludes warmups and tactical matches)
    val_eval_mask = np.array([
        (m_id not in first_match_ids) and (is_tac != 1)
        for m_id, is_tac in zip(val_match_ids, val_is_tactical)
    ])

    # -------------------- Logistic Regression (tuned) --------------------
    # Baseline tuned LR (already above) – we keep it as reference
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    log_reg = LogisticRegression(
        max_iter=1000,
        C=2.0,                # try a larger C for less regularisation
        solver='lbfgs',
        class_weight='balanced',
    )
    log_reg.fit(X_train_scaled, y_train)
    pred_log = log_reg.predict(X_val_scaled)
    acc_log = accuracy_score(y_val, pred_log)
    print(f"Logistic Regression (tuned) validation accuracy: {acc_log:.2%}")
    if np.any(val_eval_mask):
        acc_log_filtered = accuracy_score(y_val[val_eval_mask], pred_log[val_eval_mask])
        print(f"  Warm-up-excluded validation accuracy: {acc_log_filtered:.2%}")

    # -------------------- Random Forest --------------------
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_val)
    acc_rf = accuracy_score(y_val, pred_rf)
    print(f"Random Forest validation accuracy: {acc_rf:.2%}")
    if np.any(val_eval_mask):
        acc_rf_filtered = accuracy_score(y_val[val_eval_mask], pred_rf[val_eval_mask])
        print(f"  Warm-up-excluded validation accuracy: {acc_rf_filtered:.2%}")

    # -------------------- Gradient Boosting --------------------
    gb = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    pred_gb = gb.predict(X_val)
    acc_gb = accuracy_score(y_val, pred_gb)
    print(f"Gradient Boosting validation accuracy: {acc_gb:.2%}")
    if np.any(val_eval_mask):
        acc_gb_filtered = accuracy_score(y_val[val_eval_mask], pred_gb[val_eval_mask])
        print(f"  Warm-up-excluded validation accuracy: {acc_gb_filtered:.2%}")

    # -------------------- XGBoost shallow model with early stopping --------------------
    if XGB_AVAILABLE:
        xgb_model = XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=500,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            use_label_encoder=False,
        )
        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False,
        )
        pred_xgb = xgb_model.predict(X_val)
        acc_xgb = accuracy_score(y_val, pred_xgb)
        print(f"XGBoost (max_depth=3, early stop) validation accuracy: {acc_xgb:.2%}")
        if np.any(val_eval_mask):
            acc_xgb_filtered = accuracy_score(y_val[val_eval_mask], pred_xgb[val_eval_mask])
            print(f"  Warm-up-excluded validation accuracy: {acc_xgb_filtered:.2%}")
    else:
        print("XGBoost not installed – skipping XGBoost comparison.")

    # -------------------- Warm-up-excluded accuracy for each model --------------------
    from metrics import warmup_excluded_accuracy
    overall_warmup_acc = warmup_excluded_accuracy()
    print(f"Warm-up-excluded overall pipeline accuracy (current predict.py): {overall_warmup_acc:.2%}")

    # -------------------------------------------------------
    # Additional LR experiments: varying C, L1 penalty, custom class weights
    # -------------------------------------------------------
    from sklearn.utils import compute_class_weight
    import itertools
    Cs = [0.5, 1.0, 2.0, 5.0]
    classes = np.unique(y_train)
    balanced_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    balanced_dict = dict(zip(classes, balanced_weights))
    custom_dict = balanced_dict.copy()
    if 0 in custom_dict:
        custom_dict[0] *= 2.0
    weight_options = ["balanced", custom_dict]

    best = (None, None, 0.0)
    solver = "lbfgs"
    for C, w in itertools.product(Cs, weight_options):
        try:
            lr = LogisticRegression(
                max_iter=2000,
                C=C,
                solver=solver,
                class_weight=w,
            )
            # Fit using the same scaled data as the baseline LR
            lr.fit(X_train_scaled, y_train)
            pred = lr.predict(X_val_scaled)
            acc = accuracy_score(y_val, pred)
            # Track best configuration
            if acc > best[2]:
                best = (C, w, acc)
            if np.any(val_eval_mask):
                acc_fil = accuracy_score(y_val[val_eval_mask], pred[val_eval_mask])
                print(f"LR C={C}, class_weight={w} => acc: {acc:.2%} (Warm-up-excluded: {acc_fil:.2%})")
            else:
                print(f"LR C={C}, class_weight={w} => acc: {acc:.2%}")
        except Exception as e:
            print(f"Skipped LR C={C}, class_weight={w}: {e}")

    # Report best LR configuration after exploring all options
    best_C, best_w, best_acc = best
    print("\nBest Logistic Regression configuration:")
    print(f"  C = {best_C}")
    print(f"  class_weight = {best_w}")
    print(f"  validation accuracy = {best_acc:.2%}\n")

if __name__ == "__main__":
    main()
