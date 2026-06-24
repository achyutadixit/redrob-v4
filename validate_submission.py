import os
import sys
import pandas as pd

def main():
    filepath = 'outputs/team_antigravity.csv'
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        sys.exit(1)
        
    try:
        df = pd.read_csv(filepath)
        if len(df) == 0:
            print("Error: Submission file is empty.")
            sys.exit(1)
            
        required_cols = {'rank', 'candidate_id', 'score', 'reasoning'}
        if not required_cols.issubset(set(df.columns)):
            print(f"Error: Missing required columns. Expected {required_cols}, got {set(df.columns)}")
            sys.exit(1)
            
        print(f"Validation successful. {len(df)} candidates ranked.")
        sys.exit(0)
    except Exception as e:
        print(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
