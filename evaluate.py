import polars as pl
import re

def evaluate(file_path):
    df = pl.read_csv(file_path)
    # Filter for matches that have an ACTUAL result
    actual_df = df.filter(pl.col("status_or_prediction").str.starts_with("ACTUAL:"))
    
    wrong_count = 0
    total_count = actual_df.height
    wrong_matches = []

    for row in actual_df.to_dicts():
        status = row["status_or_prediction"]
        # Pattern: ACTUAL: {actual} (Model: {pred})
        match = re.search(r"ACTUAL: (.*?) \(Model: (.*?)\)", status)
        if match:
            actual_res = match.group(1).strip()
            model_pred = match.group(2).strip()
            
            # Normalize model_pred: "Draw/Tight Match (...)" -> "Draw"
            if model_pred.startswith("Draw/Tight Match"):
                norm_pred = "Draw"
            else:
                norm_pred = model_pred
            
            if actual_res != norm_pred:
                wrong_count += 1
                wrong_matches.append(f"Match {row['match_id']}: {row['team_home']} vs {row['team_away']} | Actual: {actual_res}, Pred: {model_pred}")
                
    return total_count, wrong_count, wrong_matches

if __name__ == "__main__":
    total, wrong, details = evaluate("data/predictions.csv")
    print(f"Total Matches with Results: {total}")
    print(f"Wrong Predictions: {wrong}")
    if wrong > 0:
        print("Details:")
        for d in details:
            print(f"  - {d}")
