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
    
    top_5 = form_df.head(5)
    
    # Big teams to include
    big_teams = ["Germany", "Brazil", "France", "Spain", "England", "Argentina"]
    
    # Identify big teams not in top 5
    top_5_teams = top_5["team"].to_list()
    missing_big_teams = [t for t in big_teams if t not in top_5_teams]
    
    # Filter the full form_df for these missing big teams
    missing_df = form_df.filter(pl.col("team").is_in(missing_big_teams))
    
    # Combine top 5 with missing big teams
    final_df = pl.concat([top_5, missing_df])
    
    print("\nTop 5 Teams (and key powerhouses):")
    print(final_df)

if __name__ == "__main__":
    main()
