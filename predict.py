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
    return teams, matches

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

def preprocess_matches(matches, teams, mapping):
    # Replace slots with actual teams based on the mapping
    matches = matches.with_columns([
        pl.col("team_home").replace(mapping).alias("team_home"),
        pl.col("team_away").replace(mapping).alias("team_away")
    ])
    
    # Join with team data for home
    matches = matches.join(teams, left_on="team_home", right_on="team", how="left")
    matches = matches.rename({"world_ranking": "rank_home", "appearances": "apps_home"})
    
    # Join with team data for away
    matches = matches.join(teams, left_on="team_away", right_on="team", how="left")
    matches = matches.rename({"world_ranking": "rank_away", "appearances": "apps_away"})
    
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
    
    # If one or both teams are still slots (Unknown in teams.csv)
    if rank_h is None or rank_a is None:
        return f"Pending Draw ({t_h} vs {t_a})"
    
    # Heuristic Prediction
    h_score = (200 - rank_h) + (apps_h * 5)
    a_score = (200 - rank_a) + (apps_a * 5)
    
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
    teams, matches = load_data()
    mapping = get_slot_mapping()
    
    print("Updating ELO (if results available)...")
    teams = update_elo(teams, matches)
    
    print("Preprocessing matches...")
    processed_matches = preprocess_matches(matches, teams, mapping)
    
    print("Processing results and predictions...")
    results = processed_matches.with_columns(
        pl.struct(["rank_home", "rank_away", "apps_home", "apps_away", "team_home", "team_away", "score_home", "score_away"])
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
