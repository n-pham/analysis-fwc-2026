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
    return teams, matches, friendlies

def get_slot_mapping():
    """
    Return the mapping of group slots (A2, B2, etc.) to actual team names.
    This should be updated once the final draw is announced.
    For now, we return an empty dict or a partial mapping.
    """
    return {
        # Example mapping (to be updated after draw)
        # "A2": "France",
        # "B2": "Argentina",
    }

def get_form_scores(friendlies, teams_list):
    if friendlies is None:
        return {team: 0 for team in teams_list}
    
    form = {team: 0 for team in teams_list}
    for row in friendlies.to_dicts():
        h, a, s_h, s_a = row["team_home"], row["team_away"], row["score_home"], row["score_away"]
        if h in form:
            if s_h > s_a: form[h] += 3
            elif s_h == s_a: form[h] += 1
        if a in form:
            if s_a > s_h: form[a] += 3
            elif s_a == s_h: form[a] += 1
    return form

def preprocess_matches(matches, teams, mapping, form_scores):
    # Add form scores to teams
    form_df = pl.DataFrame({
        "team": list(form_scores.keys()),
        "form_score": list(form_scores.values())
    })
    teams = teams.join(form_df, on="team", how="left")

    # Replace slots with actual teams based on the mapping
    matches = matches.with_columns([
        pl.col("team_home").replace(mapping).alias("team_home"),
        pl.col("team_away").replace(mapping).alias("team_away")
    ])
    
    # Join with team data for home
    matches = matches.join(teams, left_on="team_home", right_on="team", how="left")
    matches = matches.rename({"world_ranking": "rank_home", "appearances": "apps_home", "form_score": "form_home"})
    
    # Join with team data for away
    matches = matches.join(teams, left_on="team_away", right_on="team", how="left")
    matches = matches.rename({"world_ranking": "rank_away", "appearances": "apps_away", "form_score": "form_away"})
    
    return matches

def predict_logic(struct_row):
    # Check if match has already been played
    s_h = struct_row["score_home"]
    s_a = struct_row["score_away"]
    t_h = struct_row["team_home"]
    t_a = struct_row["team_away"]

    # If scores exist, determine actual winner
    if s_h is not None and s_a is not None:
        if s_h > s_a:
            return f"ACTUAL: {t_h}"
        elif s_a > s_h:
            return f"ACTUAL: {t_a}"
        else:
            return "ACTUAL: Draw"

    rank_h = struct_row["rank_home"]
    rank_a = struct_row["rank_away"]
    apps_h = struct_row["apps_home"]
    apps_a = struct_row["apps_away"]
    form_h = struct_row["form_home"] or 0
    form_a = struct_row["form_away"] or 0
    
    # If one or both teams are still slots (Unknown in teams.csv)
    if rank_h is None or rank_a is None:
        return f"Pending Draw ({t_h} vs {t_a})"
    
    # Heuristic Prediction
    # Power ranking (max 200) + Apps (max 100) + Form (weighted)
    # Form weight increased to 8 to better reflect recent momentum
    h_score = (200 - rank_h) + (apps_h * 5) + (form_h * 8)
    a_score = (200 - rank_a) + (apps_a * 5) + (form_a * 8)
    
    # Home advantage for hosts
    if t_h in ["Mexico", "Canada", "USA"]:
        h_score += 20
    if t_a in ["Mexico", "Canada", "USA"]:
        a_score += 20
        
    if h_score > a_score + 5:
        return t_h
    elif a_score > h_score + 5:
        return t_a
    else:
        return "Draw/Tight Match"

def update_elo(teams, matches):
    """
    Placeholder for ELO update logic.
    """
    return teams

def main():
    print("Loading data...")
    teams, matches, friendlies = load_data()
    mapping = get_slot_mapping()
    
    print("Calculating form scores from friendlies...")
    form_scores = get_form_scores(friendlies, teams["team"].to_list())
    
    print("Updating ELO (if results available)...")
    teams = update_elo(teams, matches)
    
    print("Preprocessing matches...")
    processed_matches = preprocess_matches(matches, teams, mapping, form_scores)
    
    print("Processing results and predictions...")
    results = processed_matches.with_columns(
        pl.struct(["rank_home", "rank_away", "apps_home", "apps_away", "form_home", "form_away", "team_home", "team_away", "score_home", "score_away"])
        .map_elements(predict_logic, return_dtype=pl.String)
        .alias("status_or_prediction")
    )
    
    print("\nResults/Predictions (Sample):")
    # Show host matches which have known teams
    host_matches = results.filter(pl.col("team_home").is_in(["Mexico", "Canada", "USA"]) | pl.col("team_away").is_in(["Mexico", "Canada", "USA"]))
    print(host_matches.select(["match_id", "date", "team_home", "team_away", "status_or_prediction"]).head(10))
    
    results.write_csv("data/predictions.csv")
    print("\nFull output saved to data/predictions.csv")

if __name__ == "__main__":
    main()
