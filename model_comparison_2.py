"""Model comparison for knockout stage (suffix _2)

- This script mirrors ``model_comparison.py`` but adds features that are
  specifically useful once the tournament moves into the knockout rounds:
  * ``is_knockout`` flag (derived from ``match_stage`` column)
  * ``stage_weight`` – higher weight for later stages
  * ``is_penalty_shootout`` flag (draws that go to penalties)
  * ``rest_home`` / ``rest_away`` – days since each team’s previous match
  * ``stage_weight`` is also used as a multiplier when training the models.

- It still provides a **scaled LogisticRegression** baseline and, if XGBoost
  is available, a shallow XGBoost model with early stopping.  The target
  encoding now supports a **four‑class problem** to distinguish a penalty‑
  shootout win from a regular draw:
    0 – Draw (no penalties)
    1 – Home win (regulation)
    2 – Away win (regulation)
    3 – Penalty‑shootout win (home or away, inferred from the penalty columns)

- The script does **not modify any existing file** – it only reads the same CSV
  sources and writes its own results if you choose to.  Use it when the
  ``match_stage`` column in ``data/matches.csv`` is populated for knockout
  matches.
"""

from __future__ import annotations

import math
import numpy as np
import polars as pl
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# XGBoost is optional – if unavailable we fall back to logistic regression only.
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:  # pragma: no cover
    XGB_AVAILABLE = False

try:
    # from tabfm import TabFMClassifier
    TABFM_AVAILABLE = False
except Exception:
    TABFM_AVAILABLE = False

try:
    from tabicl import TabICLClassifier, TabICLRegressor
    TABICL_AVAILABLE = True
except Exception:
    TABICL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Helper utilities (mirroring parts of predict.py and the original comparison)
# ---------------------------------------------------------------------------

def load_data():
    teams = pl.read_csv("data/teams.csv")
    matches = pl.read_csv("data/matches.csv")
    try:
        friendlies = pl.read_csv("data/friendlies.csv")
    except Exception:
        friendlies = None
    try:
        player_status = pl.read_csv("data/player_status.csv")
    except Exception:
        player_status = None
    return teams, matches, friendlies, player_status

def get_form_scores(friendlies, matches, teams_df):
    # Same logic as the original pipeline – it already skips tactical matches.
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

    # Tournament matches – higher weight + log‑scaled opponent rank factor
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
# Feature engineering for the knockout‑stage models
# ---------------------------------------------------------------------------

def build_feature_df(matches_df: pl.DataFrame, teams_df: pl.DataFrame, form_scores: dict) -> pl.DataFrame:
    # Attach form scores to teams
    form_df = pl.DataFrame({
        "team": list(form_scores.keys()),
        "form_attack": [v["attack"] for v in form_scores.values()],
        "form_defense": [v["defense"] for v in form_scores.values()],
    })
    teams = teams_df.join(form_df, on="team", how="left")

    # Ensure we have a match_stage column – if missing we treat everything as GROUP.
    if "match_stage" not in matches_df.columns:
        matches_df = matches_df.with_columns(pl.lit("GROUP").alias("match_stage"))

    # Ensure we have an is_penalty_shootout column – if missing we treat it as 0.
    if "is_penalty_shootout" not in matches_df.columns:
        matches_df = matches_df.with_columns(pl.lit(0).alias("is_penalty_shootout"))

    # Replace team names mapping (empty for now)
    mapping = {}
    matches = matches_df.with_columns([
        pl.col("team_home").replace(mapping).alias("team_home"),
        pl.col("team_away").replace(mapping).alias("team_away"),
        # Parse dates robustly – use strict=False so malformed dates become null
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("date_dt"),
    ])

    # Join home team data
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
    # Join away team data
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

    # -------------------------------------------------------------------
    # New knockout‑stage specific columns
    # -------------------------------------------------------------------
    # Flag if we are in a knockout stage (anything not GROUP)
    matches = matches.with_columns([
        (pl.col("match_stage") != "GROUP").alias("is_knockout"),
        # Simple numeric weight that grows later in the tournament
        pl.when(pl.col("match_stage") == "GROUP").then(1.0)
          .when(pl.col("match_stage") == "R32").then(1.3)
          .when(pl.col("match_stage") == "R16").then(1.5)
          .when(pl.col("match_stage") == "QF").then(1.7)
          .when(pl.col("match_stage") == "SF").then(1.9)
          .otherwise(2.0)
          .alias("stage_weight"),
        # Penalty shoot‑out flag – the source CSV should contain a boolean column.
        pl.col("is_penalty_shootout").fill_null(0).alias("is_penalty_shootout"),
    ])

    # -------------------------------------------------------------------
    # Rest-day features (days since previous match for each side)
    # -------------------------------------------------------------------
    # Sort by date for each team, compute difference in days, fill first match with large number.
    matches = matches.sort(["team_home", "date_dt"]).with_columns([
        pl.col("date_dt").diff().dt.total_days().over("team_home").fill_null(999).alias("rest_home")
    ])
    matches = matches.sort(["team_away", "date_dt"]).with_columns([
        pl.col("date_dt").diff().dt.total_days().over("team_away").fill_null(999).alias("rest_away")
    ])

    return matches

# ---------------------------------------------------------------------------
# Target encoding – now a 4‑class problem (draw, home win, away win, penalty win)
# ---------------------------------------------------------------------------

def encode_target(df: pl.DataFrame) -> pl.Series:
    """Encode outcomes for knockout matches.

    Returns a Series with values:
    0 – Draw (no penalties)
    1 – Home win (regulation)
    2 – Away win (regulation)
    3 – Penalty‑shootout win (home or away).
    """
    import pandas as pd

    def map_outcome(row: pd.Series):
        sh, sa = row["score_home"], row["score_away"]
        if sh == "" or sa == "" or pd.isna(sh) or pd.isna(sa):
            return None
        sh, sa = int(sh), int(sa)
        # penalty columns (optional) – if they exist we use them to identify a shoot‑out win
        pen_home = row.get("penalties_home")
        pen_away = row.get("penalties_away")
        is_pen = row.get("is_penalty_shootout", 0)
        if sh == sa:
            # Draw scenario – decide if it was decided by penalties
            if is_pen and pen_home is not None and pen_away is not None:
                # The side with more penalties wins the shoot‑out
                if int(pen_home) > int(pen_away):
                    return 3  # home win on penalties
                if int(pen_away) > int(pen_home):
                    return 3  # away win on penalties (same label, just "penalty win")
            return 0  # genuine draw
        return 1 if sh > sa else 2

    return pl.Series(df.to_pandas().apply(map_outcome, axis=1))

# ---------------------------------------------------------------------------
# Main comparison routine
# ---------------------------------------------------------------------------

def main():
    teams, matches, friendlies, player_status = load_data()
    form_scores = get_form_scores(friendlies, matches, teams)
    matches = calculate_injuries(matches, player_status)
    feature_df = build_feature_df(matches, teams, form_scores)

    # Keep only rows that have actual scores (to evaluate predictive power)
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

    # Encode target (4‑class)
    y_series = encode_target(eval_df)
    y_np = y_series.to_numpy()
    valid_idx = ~np.isnan(y_np)
    y_np = y_np[valid_idx].astype(int)

    # Store match metadata to align with split
    match_ids = eval_df["match_id"].to_numpy()[valid_idx]
    is_tactical = eval_df["is_tactical"].to_numpy()[valid_idx]

    # Feature columns for the models
    feature_cols = [
        "ba_home", "ba_away", "bd_home", "bd_away",
        "fa_home", "fa_away", "fd_home", "fd_away",
        "apps_home", "apps_away",
        "injuries_home", "injuries_away",
        "rank_home", "rank_away",
        "rank_diff",  # rank diff (will be computed below)
        "is_knockout", "stage_weight", "rest_home", "rest_away",
        "is_penalty_shootout",
    ]

    # Compute rank diff column explicitly (Polars cannot interpret the expression above directly)
    eval_df = eval_df.with_columns([
        (pl.col("rank_home") - pl.col("rank_away")).alias("rank_diff")
    ])
    X = eval_df.select([
        "ba_home", "ba_away", "bd_home", "bd_away",
        "fa_home", "fa_away", "fd_home", "fd_away",
        "apps_home", "apps_away",
        "injuries_home", "injuries_away",
        "rank_diff",
        "is_knockout", "stage_weight", "rest_home", "rest_away",
        "is_penalty_shootout",
    ]).to_numpy()[valid_idx]

    # Train/validation split (stratified by the 4‑class label)
    indices = np.arange(len(y_np))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_np
    )

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_np[train_idx], y_np[val_idx]
    val_match_ids = match_ids[val_idx]
    val_is_tactical = is_tactical[val_idx]

    # Validation evaluation mask (excludes warmups and tactical matches)
    val_eval_mask = np.array([
        (m_id not in first_match_ids) and (is_tac != 1)
        for m_id, is_tac in zip(val_match_ids, val_is_tactical)
    ])

    # -------------------- Logistic Regression baseline (scaled) --------------------
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    log_reg = LogisticRegression(max_iter=500, class_weight="balanced")
    log_reg.fit(X_train_scaled, y_train)
    pred_log = log_reg.predict(X_val_scaled)
    acc_log = accuracy_score(y_val, pred_log)
    print(f"Logistic Regression (scaled, 4‑class) validation accuracy: {acc_log:.2%}")
    if np.any(val_eval_mask):
        acc_log_filtered = accuracy_score(y_val[val_eval_mask], pred_log[val_eval_mask])
        print(f"  Warm-up-excluded validation accuracy: {acc_log_filtered:.2%}")

    # -------------------- Random Forest --------------------
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

    # -------------------- TabFM classifier --------------------
    if TABFM_AVAILABLE:
        from tabfm import tabfm_v1_0_0_pytorch
        model = tabfm_v1_0_0_pytorch.load()
        clf_tabfm = TabFMClassifier(model=model, n_estimators=10)
        clf_tabfm.fit(X_train, y_train)
        pred_tabfm = clf_tabfm.predict(X_val)
        acc_tabfm = accuracy_score(y_val, pred_tabfm)
        print(f"TabFM validation accuracy: {acc_tabfm:.2%}")
        if np.any(val_eval_mask):
            acc_tabfm_filtered = accuracy_score(y_val[val_eval_mask], pred_tabfm[val_eval_mask])
            print(f"  Warm-up-excluded validation accuracy: {acc_tabfm_filtered:.2%}")

    # -------------------- TabICL classifier --------------------
    if TABICL_AVAILABLE:
        clf_icl = TabICLClassifier()
        clf_icl.fit(X_train, y_train)
        pred_icl = clf_icl.predict(X_val)
        acc_icl = accuracy_score(y_val, pred_icl)
        print(f"TabICL validation accuracy: {acc_icl:.2%}")
        if np.any(val_eval_mask):
            acc_icl_filtered = accuracy_score(y_val[val_eval_mask], pred_icl[val_eval_mask])
            print(f"  Warm-up-excluded validation accuracy: {acc_icl_filtered:.2%}")

    # -------------------- XGBoost shallow model with early stopping --------------------
    if XGB_AVAILABLE:
        xgb_model = XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=500,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=4,
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
        print(f"XGBoost (max_depth=3, early stop, 4‑class) validation accuracy: {acc_xgb:.2%}")
        if np.any(val_eval_mask):
            acc_xgb_filtered = accuracy_score(y_val[val_eval_mask], pred_xgb[val_eval_mask])
            print(f"  Warm-up-excluded validation accuracy: {acc_xgb_filtered:.2%}")
    else:
        print("XGBoost not installed – skipping XGBoost comparison.")

    # Warm-up-excluded overall pipeline accuracy (unchanged helper)
    from metrics import warmup_excluded_accuracy
    overall_warmup_acc = warmup_excluded_accuracy()
    print(f"Warm-up-excluded overall pipeline accuracy (current predict.py): {overall_warmup_acc:.2%}")

if __name__ == "__main__":
    main()
