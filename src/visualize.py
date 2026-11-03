import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

"""Generates figures for the report from raw_results.csv and games_howell_results.csv. Writes to figures/."""

# box plot of training size by accuracy
df = pd.read_csv("../data/raw_results.csv")

# Figure 1
plot_data = df[['train_size', 'accuracy']]

sns.boxplot(
    data = plot_data,
    x='train_size',
    y='accuracy',
    hue='train_size',
    palette='Blues_r',
    width=0.6    
)
plt.title("Model Accuracy Distribution by Training Set Size", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Training Dataset Size", fontsize=12, fontweight='bold')
plt.ylabel("Model Accuracy Score", fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.5) # Soft background lines to guide the eye
plt.tight_layout()
plt.savefig('../figures/train_size_vs_accuracy.png', dpi=150)
plt.close()

# Figure 2
sns.lineplot(data=df, x='train_size', y='accuracy', marker='o')
plt.title("Mean Accuracy by Training Set Size\nGains diminish with each doubling of training data", fontsize=13, fontweight='bold')
plt.xlabel("Training Dataset Size", fontsize=11, fontweight='bold')
plt.ylabel("Mean Model Accuracy", fontsize=11, fontweight='bold')
plt.savefig('../figures/mean_accuracy_trend.png', dpi=150)
plt.close()

# Figure 3
gh = pd.read_csv("../data/games_howell_results.csv")
gh['pair'] = gh['A'].astype(str) + " vs " + gh['B'].astype(str)

plt.barh(gh['pair'], gh['diff'], xerr=2 * gh['se'])
plt.axvline(0, color='black', linestyle='--')
plt.title("Pairwise Mean Differences in Accuracy")
plt.xlabel("Difference in Mean Accuracy\n(negative values indicate the larger training set performed better)")
plt.ylabel("Comparison")
plt.tight_layout()
plt.savefig('../figures/pairwise_differences.png', dpi=150)
plt.close()


# Figure 4
sns.scatterplot(data=df, x='train_time', y='accuracy', hue='train_size', palette='viridis')
plt.title("Accuracy vs Training Time")
plt.xlabel("Training Time (seconds)")
plt.ylabel("Model Accuracy")
plt.tight_layout()
plt.savefig('../figures/accuracy_vs_time.png', dpi=150)
plt.close()

# Figure 5 of QQ plot comes from normality confirmation in analyze.py
