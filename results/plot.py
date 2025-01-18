import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data with headers
df = pd.read_csv("average_scores.csv")

# Extract token count from model names
df["token_count"] = df["model"].apply(lambda x: int(x.split("token")[1].split("B")[0]))
# Remove token count 41
df = df[df["token_count"] != 41]

# Define learning rates and titles for filtering and plotting
learning_rates = {
    "-1e-3pt-4e-5ft": (1, 1, 'LR 4e-5'),
    "-1e-3pt-8e-6ft": (0, 0, 'LR 8e-6'),
    "-1e-3pt-1e-5ft": (0, 1, 'LR 1e-5')
}

metric_list = ['score_1', 'score_2', 'score_3', 'score_4', 'score_5', 'average_score']
metric_names = ["pope.csv", "textvqa.csv", "gqa.csv", "mme_Perception.csv", "mme_Cognition.csv", "average_score"]

for kk, metric in enumerate(metric_list):
    # Create a subplot figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    #main title as "Average VLM Perf across 5 benchmarks"
    fig.suptitle(f"Average VLM Perf across 5 benchmarks", fontsize=20)


    # Plot for each learning rate
    for lr, (row, col, title) in learning_rates.items():
        df_lr = df[df["model"].str.contains(lr)].sort_values(by='token_count')
        axes[row, col].plot(df_lr["token_count"], df_lr[metric], 'o-', label=lr)
        axes[row, col].set_xlabel('Token Count')
        axes[row, col].set_ylabel('Average Score')
        axes[row, col].set_title(title)
        axes[row, col].grid(True)

    # Plot for models with '-finetune' but no specific LR mentioned
    df_finetune = df[df["model"].str.contains("-finetune") & ~df["model"].str.contains("-1e-3pt")]
    df_finetune = df_finetune.sort_values(by='token_count')
    axes[1, 0].plot(df_finetune["token_count"], df_finetune[metric], 'o-', label='LR 2e-5')
    axes[1, 0].set_xlabel('Token Count')
    axes[1, 0].set_ylabel('Average Score')
    axes[1, 0].set_title('LR 2e-5')
    axes[1, 0].grid(True)

    # Adjust layout and save the plot
    plt.tight_layout()

    plt.savefig(f'plots/zeroshot_evals_agg_{metric_names[kk]}.png')
