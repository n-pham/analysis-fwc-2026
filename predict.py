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
    rank_map = {row["team"]: row["world_ranking"] for row in teams_df.to_dicts()}
    form = {team: 0 for team in teams_list}
    
    # Scores from friendlies
    if friendlies is not None:
        for row in friendlies.to_dicts():
            h, a, s_h, s_a = row["team_home"], row["team_away"], row["score_home"], row["score_away"]
            if h in form and s_h is not None and s_a is not None:
                if s_h > s_a: form[h] += 3
                elif s_h == s_a: form[h] += 1
            if a in form and s_h is not None and s_a is not None:
                if s_a > s_h: form[a] += 3
                elif s_a == s_h: form[a] += 1
                
    # Scores from actual tournament matches (Higher weight for tournament form)
    for row in matches.to_dicts():
        h, a, s_h, s_a = row["team_home"], row["team_away"], row["score_home"], row["score_away"]
        if s_h is not None and s_a is not None:
            weight = 10 
            r_h = rank_map.get(h, 100)
            r_a = rank_map.get(a, 100)
            gd = s_h - s_a

            # Home team calculation
            if h in form:
                pts_h = 0
                if s_h > s_a: 
                    pts_h = 5 # Base win
                    if r_a <= 15: pts_h += 5 # Elite opponent win bonus
                    if r_a > 75: pts_h -= 1 # Reduced Minnow win adjustment
                    if r_a < r_h: pts_h += (r_h - r_a) / 10.0 # Reduced underdog bonus
                    if s_a == 0: pts_h += 1.0 # Reduced Clean sheet bonus
                    if gd >= 3: pts_h += gd / 2.0 # Scaled Dominance bonus (rewarding big wins)
                elif s_h == s_a: 
                    pts_h = 2 # Base draw
                    if r_a <= 15: pts_h += 5 # Elite opponent draw bonus
                    if r_a < r_h: pts_h += (r_h - r_a) / 10.0 
                else: 
                    pts_h = -1
                    if gd <= -3: pts_h -= abs(gd) / 2.0 
                form[h] += pts_h * weight

            # Away team calculation
            if a in form:
                pts_a = 0
                if s_a > s_h: 
                    pts_a = 5
                    if r_h <= 15: pts_a += 5 # Elite opponent win bonus
                    if r_h > 75: pts_a -= 1 # Reduced Minnow win adjustment
                    if r_h < r_a: pts_a += (r_a - r_h) / 10.0
                    if s_h == 0: pts_a += 1.0 # Reduced Clean sheet bonus
                    if gd <= -3: pts_a += abs(gd) / 2.0
                elif s_a == s_h: 
                    pts_a = 2
                    if r_h <= 15: pts_a += 5 # Elite opponent draw bonus
                    if r_h < r_a: pts_a += (r_a - r_h) / 10.0
                else: 
                    pts_a = -1
                    if gd >= 3: pts_a -= gd / 2.0
                form[a] += pts_a * weight

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
        "form_score": [float(v) for v in form_scores.values()]
    })
    teams = teams.join(form_df, on="team", how="left")

    # Replace slots
    matches = matches.with_columns([
        pl.col("team_home").replace(mapping).alias("team_home"),
        pl.col("team_away").replace(mapping).alias("team_away")
    ])
    
    # Join home
    matches = matches.join(teams, left_on="team_home", right_on="team", how="left")
    matches = matches.rename({"world_ranking": "rank_home", "appearances": "apps_home", "form_score": "form_home"})
    
    # Join away
    matches = matches.join(teams, left_on="team_away", right_on="team", how="left")
    matches = matches.rename({"world_ranking": "rank_away", "appearances": "apps_away", "form_score": "form_away"})
    
    return matches

def predict_logic(struct_row):
    s_h = struct_row["score_home"]
    s_a = struct_row["score_away"]
    t_h = struct_row["team_home"]
    t_a = struct_row["team_away"]

    if s_h is not None and s_a is not None:
        if s_h > s_a: return f"ACTUAL: {t_h}"
        elif s_a > s_h: return f"ACTUAL: {t_a}"
        else: return "ACTUAL: Draw"

    rank_h, rank_a = struct_row["rank_home"], struct_row["rank_away"]
    apps_h, apps_a = struct_row["apps_home"], struct_row["apps_away"]
    form_h, form_a = struct_row["form_home"] or 0, struct_row["form_away"] or 0
    inj_h, inj_a = struct_row["injuries_home"], struct_row["injuries_away"]
    
    if rank_h is None or rank_a is None:
        return f"Pending Draw ({t_h} vs {t_a})"
    
    h_score = (200 - rank_h) + (apps_h * 12) + (form_h * 5) - (inj_h * 15)
    a_score = (200 - rank_a) + (apps_a * 12) + (form_a * 5) - (inj_a * 15)
    
    if t_h in ["Mexico", "Canada", "USA"]: h_score += 20
    if t_a in ["Mexico", "Canada", "USA"]: a_score += 20
        
    if h_score > a_score + 15: return t_h
    elif a_score > h_score + 15: return t_a
    else: return "Draw/Tight Match"

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
        "rank_home", "rank_away",
        "apps_home", "apps_away",
        "form_home", "form_away",
        "injuries_home", "injuries_away",
        "status_or_prediction"
    ]
    
    # Ensure all columns exist before selecting (safety check)
    final_cols = [c for c in ordered_columns if c in results.columns]
    results = results.select(final_cols)
    
    print("\nResults/Predictions (Sample with injuries):")
    cols_to_show = ["match_id", "team_home", "team_away", "injuries_home", "injuries_away", "status_or_prediction"]
    print(results.filter(pl.col("match_id").is_in([1, 25, 28])).select(cols_to_show))
    
    results.write_csv("data/predictions.csv")
    print("\nFull output saved to data/predictions.csv (with ordered columns)")

if __name__ == "__main__":
    main()
