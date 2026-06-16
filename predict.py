import polars as pl
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
import numpy as np
from sklearn.model_selection import train_test_split

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

def get_slot_mapping():
    """
    Return the mapping of group slots (A2, B2, etc.) to actual team names.
    """
    return {}

def get_form_scores(friendlies, matches, teams_df):
    teams_list = teams_df["team"].to_list()
    # Initialize separate attack and defense form
    form = {team: {"attack": 0.0, "defense": 0.0} for team in teams_list}
    
    # Scores from friendlies (Low weight)
    if friendlies is not None:
        for row in friendlies.to_dicts():
            h, a, s_h, s_a = row["team_home"], row["team_away"], row["score_home"], row["score_away"]
            if s_h is None or s_a is None: continue
            
            if h in form:
                form[h]["attack"] += s_h * 1.0
                if s_a == 0: form[h]["defense"] += 2.0
            if a in form:
                form[a]["attack"] += s_a * 1.0
                if s_h == 0: form[a]["defense"] += 2.0
                
    # Scores from actual tournament matches (High weight)
    for row in matches.to_dicts():
        h, a, s_h, s_a = row["team_home"], row["team_away"], row["score_home"], row["score_away"]
        if s_h is not None and s_a is not None:
            # Home team
            if h in form:
                form[h]["attack"] += s_h * 5.0
                if s_a == 0: form[h]["defense"] += 10.0
                form[h]["defense"] -= s_a * 3.0
            
            # Away team
            if a in form:
                form[a]["attack"] += s_a * 5.0
                if s_h == 0: form[a]["defense"] += 10.0
                form[a]["defense"] -= s_h * 3.0

    return form

def calculate_injuries(matches, player_status):
    """
    Calculates injuries/suspensions for each match based on player_status.
    """
    inj_h_list = []
    inj_a_list = []
    
    for row in matches.to_dicts():
        m_id = row["match_id"]
        t_h = row["team_home"]
        t_a = row["team_away"]
        
        inj_h = 0
        inj_a = 0
        
        if player_status is not None:
            # Home
            h_out = player_status.filter((pl.col("team") == t_h) & (pl.col("is_key") == True))
            for p in h_out.to_dicts():
                unavail = [u.strip() for u in str(p["unavailable_match_ids"]).split(",")]
                if str(m_id) in unavail:
                    inj_h += 1
            # Away
            a_out = player_status.filter((pl.col("team") == t_a) & (pl.col("is_key") == True))
            for p in a_out.to_dicts():
                unavail = [u.strip() for u in str(p["unavailable_match_ids"]).split(",")]
                if str(m_id) in unavail:
                    inj_a += 1
                    
        inj_h_list.append(inj_h)
        inj_a_list.append(inj_a)
        
    return matches.with_columns([
        pl.Series(name="injuries_home", values=inj_h_list),
        pl.Series(name="injuries_away", values=inj_a_list)
    ])

def preprocess_matches(matches, teams, mapping, form_scores):
    # Add form scores to teams
    form_df = pl.DataFrame({
        "team": list(form_scores.keys()),
        "form_attack": [float(v["attack"]) for v in form_scores.values()],
        "form_defense": [float(v["defense"]) for v in form_scores.values()]
    })
    teams = teams.join(form_df, on="team", how="left")

    # Replace slots
    matches = matches.with_columns([
        pl.col("team_home").replace(mapping).alias("team_home"),
        pl.col("team_away").replace(mapping).alias("team_away")
    ])
    
    # Join home
    matches = matches.join(teams, left_on="team_home", right_on="team", how="left")
    matches = matches.rename({
        "world_ranking": "rank_home", 
        "appearances": "apps_home", 
        "base_attack": "ba_home",
        "base_defense": "bd_home",
        "form_attack": "fa_home",
        "form_defense": "fd_home"
    })
    
    # Join away
    matches = matches.join(teams, left_on="team_away", right_on="team", how="left")
    matches = matches.rename({
        "world_ranking": "rank_away", 
        "appearances": "apps_away",
        "base_attack": "ba_away",
        "base_defense": "bd_away",
        "form_attack": "fa_away",
        "form_defense": "fd_away"
    })
    
    return matches

def predict_logic(struct_row):
    s_h = struct_row["score_home"]
    s_a = struct_row["score_away"]
    t_h = struct_row["team_home"]
    t_a = struct_row["team_away"]

    # Base Metrics
    ba_h, bd_h = struct_row["ba_home"], struct_row["bd_home"]
    ba_a, bd_a = struct_row["ba_away"], struct_row["bd_away"]
    
    # Form Metrics
    fa_h, fd_h = struct_row["fa_home"] or 0, struct_row["fd_home"] or 0
    fa_a, fd_a = struct_row["fa_away"] or 0, struct_row["fd_away"] or 0
    
    # Pedigree (Appearances)
    apps_h, apps_a = struct_row["apps_home"] or 0, struct_row["apps_away"] or 0
    
    # Injuries
    inj_h, inj_a = struct_row["injuries_home"], struct_row["injuries_away"]
    
    if ba_h is None or ba_a is None:
        pred = f"Pending Draw ({t_h} vs {t_a})"
    else:
        # Calculate Effective Attack and Defense
        # Formula: Base + Form + Pedigree - (Injuries * 10)
        h_atk = ba_h + fa_h + apps_h - (inj_h * 10)
        h_def = bd_h + fd_h + apps_h - (inj_h * 10)
        
        a_atk = ba_a + fa_a + apps_a - (inj_a * 10)
        a_def = bd_a + fd_a + apps_a - (inj_a * 10)
        
        # Host Bonus
        if t_h in ["Mexico", "Canada", "USA"]:
            h_atk += 10
            h_def += 10
        if t_a in ["Mexico", "Canada", "USA"]:
            a_atk += 10
            a_def += 10

        # Matchup Dynamics
        # How well does Home Attack break through Away Defense? (Clipped at 0)
        h_scoring_potential = max(0, h_atk - a_def)
        # How well does Away Attack break through Home Defense? (Clipped at 0)
        a_scoring_potential = max(0, a_atk - h_def)
        
        diff = h_scoring_potential - a_scoring_potential
            
        # Margin for Draw
        # A lower margin reflects that it takes less dominance to predict a winner
        if diff > 20: pred = t_h
        elif diff < -20: pred = t_a
        else: pred = "Draw/Tight Match"

    if s_h is not None and s_a is not None:
        if s_h > s_a: actual = t_h
        elif s_a > s_h: actual = t_a
        else: actual = "Draw"
        return f"ACTUAL: {actual} (Model: {pred})"
    
    return pred

def main():
    print("Loading data...")
    teams, matches, friendlies, player_status = load_data()
    mapping = get_slot_mapping()
    
    print("Calculating form scores (Friendlies + Tournament)...")
    form_scores = get_form_scores(friendlies, matches, teams)
    
    print("Calculating dynamic injuries...")
    matches_with_injuries = calculate_injuries(matches, player_status)
    
    print("Preprocessing matches...")
    processed_matches = preprocess_matches(matches_with_injuries, teams, mapping, form_scores)
    
    print("Processing results and predictions...")
    results = processed_matches.with_columns(
        pl.struct(processed_matches.columns)
        .map_elements(predict_logic, return_dtype=pl.String)
        .alias("status_or_prediction")
    )
    
    # Define desired column order: metadata, scores, then paired metrics
    ordered_columns = [
        "match_id", "date", "team_home", "team_away", "venue",
        "score_home", "score_away",
        "ba_home", "ba_away",
        "bd_home", "bd_away",
        "fa_home", "fa_away",
        "fd_home", "fd_away",
        "apps_home", "apps_away",
        "injuries_home", "injuries_away",
        "status_or_prediction"
    ]
    
    # Ensure all columns exist before selecting (safety check)
    final_cols = [c for c in ordered_columns if c in results.columns]
    results = results.select(final_cols)
    
    print("\nResults/Predictions Verification (Historical & Future):")
    cols_to_show = ["match_id", "team_home", "team_away", "status_or_prediction"]
    # Show first few matches (historical) and some future ones
    print(results.filter(pl.col("match_id").is_in([1, 6, 11, 15, 17, 39])).select(cols_to_show))
    
    results.write_csv("data/predictions.csv")
    print("\nFull output saved to data/predictions.csv (with ordered columns)")

if __name__ == "__main__":
    main()
