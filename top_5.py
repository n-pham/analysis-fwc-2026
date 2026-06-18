import polars as pl
from predict import load_data, get_form_scores

def main():
    print("Loading data...")
    teams, matches, friendlies, player_status = load_data()
    
    print("Calculating form scores...")
    form_scores = get_form_scores(friendlies, matches, teams)
    
    # Calculate total points
    form_df = pl.DataFrame({
        "team": list(form_scores.keys()),
        "points": [v["attack"] + v["defense"] for v in form_scores.values()]
    }).sort("points", descending=True)
    
    print("\nTop 5 Teams by Form Points:")
    print(form_df.head(5))

if __name__ == "__main__":
    main()
