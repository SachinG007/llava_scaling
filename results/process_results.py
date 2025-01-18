import pandas as pd

# Load the CSV files without headers
file_paths = ["pope.csv", "textvqa.csv", "gqa.csv", "mme_Perception.csv", "mme_Cognition.csv"]
# file_paths = ["pope.csv"]
dfs = [pd.read_csv(file, header=None, names=["model", f"score_{i+1}"]) for i, file in enumerate(file_paths)]

# Merge dataframes on the 'model' column
merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = merged_df.merge(df, on="model")

# Calculate the average score
score_cols = [col for col in merged_df.columns if "score" in col]
merged_df["average_score"] = merged_df[score_cols].mean(axis=1)

# Save the results to a new CSV file including all scores and the average
merged_df.to_csv("average_scores.csv", index=False, header=True)
