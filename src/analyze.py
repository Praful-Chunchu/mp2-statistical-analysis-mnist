import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pingouin as pg
import seaborn as sns
from scipy.stats import t

"""Checks ANOVA assumptions, runs Welch's ANOVA and Games-Howell post hoc. Writes data/games_howell_results.csv and figures/qq_plots.png."""

df = pd.read_csv("../data/raw_results.csv")

train_size_grouped = df.groupby('train_size')['accuracy'].agg(['mean', 'std'])
sd = train_size_grouped['std']
drop = (sd.iloc[0] - sd.iloc[-1]) / sd.iloc[0]
print(f"SD reduction from smallest to largest condition: {drop:.1%}")

print(train_size_grouped)

# We now have mean and standard deviation values for the 5 groups
# From the eye test we can see as the train_size gets larger we have an increasing mean accuracy score
# We can also see that we have a smaller and smaller standard deviation as the train_size gets larger.  
# This suggests that the bell curve is narrower.  

# Let's check for Normality here with Shapiro-Wilk's test and Q-Q plots

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, size in enumerate([500, 1000, 2000, 4000, 8000]):
    accuracy_data  = df.loc[df['train_size'] == size, 'accuracy']
    shapiro_result = stats.shapiro(accuracy_data)
    print(f"Normality check with Shapiro with train_size = {size}: " + str(shapiro_result.statistic))
    p_value_decimal = f"{shapiro_result.pvalue:.3f}".rstrip('0').rstrip('.')
    print(f"P value for Normality with Shapiro with train_size = {size}: " + str(p_value_decimal))
    sm.qqplot(accuracy_data, line='45', fit=True, ax=axes[i])

    axes[i].set_title(f"Size: {size}\np-val: {p_value_decimal}")
fig.delaxes(axes[5])
plt.tight_layout()
plt.savefig('../figures/qq_plots.png', dpi=150)

# From the p-values and confirming with Q-Q plots - we fail to reject normality thus we can assume the accuracy data is normal.


# Let's check for Equal variances here with Levene's test
group_1 = df.loc[df['train_size'] == 500, 'accuracy']
group_2 = df.loc[df['train_size'] == 1000, 'accuracy']
group_3 = df.loc[df['train_size'] == 2000, 'accuracy']
group_4 = df.loc[df['train_size'] == 4000, 'accuracy']
group_5 = df.loc[df['train_size'] == 8000, 'accuracy']

stat, p_value = stats.levene(group_1, group_2, group_3, group_4, group_5)

print(f"Levene's Statistic: {stat:.4f}")
print(f"Levene's p-value:   {p_value:.4e}")

# We find that our data is unequal in variance, thus we'll have to choose between Non-parametric tests (ranking based) or Non-parametric ANOVA
# Since our data is already normal, converting to non-parametric tests might reduce our statistical power since we throw-away numeric data 
# and convert them to ranks.

welch_result = pg.welch_anova(data=df, dv='accuracy', between='train_size')

print("Welch's results: ")
print(welch_result)

# Results:        Source     ddof1       ddof2           F         p-unc       np2
#               train_size      4    120.369006  828.492461  1.524699e-86  0.938025

gh_result = pg.pairwise_gameshowell(data=df, dv='accuracy', between='train_size')

print("Post-HOC result with Games-Howell for Welch's ANOVA: ")
print(gh_result)

margin = t.ppf(0.975, gh_result['df']) * gh_result['se']
gh_result['ci_low']  = gh_result['diff'] - margin
gh_result['ci_high'] = gh_result['diff'] + margin
gh_result.to_csv("../data/games_howell_results.csv", index=False)
print(df.groupby('train_size')['train_time'].mean())
