# Does Training Set Size Affect Model Accuracy?

**Module Project 2: Statistical Analysis**

A controlled experiment testing whether the number of training examples affects classification accuracy, using MNIST handwritten digits and logistic regression.

---

## Research Question

Does increasing the size of the training set improve model classification accuracy, and if so, by how much?

**Null hypothesis (H0):** Training set size has no effect on model classification accuracy.

**Alternative hypothesis (H1):** Increasing training set size increases mean classification accuracy.

---

## Experimental Design

| | |
|---|---|
| **Independent variable** | Training set size (500, 1000, 2000, 4000, 8000) |
| **Dependent variable** | Classification accuracy on a held-out test set |
| **Replications** | 50 per condition (250 training runs total) |
| **Dataset** | MNIST handwritten digits |
| **Model** | Logistic regression, `max_iter=1000` |

**Controlled variables:** model type and configuration, preprocessing pipeline (images flattened from 28x28 to 784-dimensional vectors, pixel values normalized by 255), and the test set (all 10,000 test images, identical for every run).

**Randomization:** each replication draws a fresh random subset of training images from the full 60,000-image pool. This means accuracy varies across replications within a condition, which is the source of variation the statistical tests operate on.

---

## Power Analysis

| Parameter | Value | Justification |
|---|---|---|
| Significance level (alpha) | 0.05 | Field convention |
| Statistical power | 0.80 | Field convention |
| Effect size (Cohen's d) | 0.5 | Medium effect, Cohen's conventions |
| **Required n per group** | **~50.15** | Output of the analysis |

**Software:** `statsmodels.stats.power.TTestIndPower`

**Scope note:** course material covered power analysis for two-group comparisons using Cohen's d. Because this design has five conditions, the power analysis was conducted on the largest expected contrast (500 vs 8000), and the resulting replication count was applied uniformly across all conditions.

**Why these values:** a stricter alpha reduces false positives but costs power. Higher power is achievable but expensive, raising power to 0.99 increases the required sample roughly tenfold. Medium effect size was chosen rather than large because assuming a large effect purely to reduce sample size produces an underpowered study if the true effect is smaller.

---

## Statistical Approach

### Assumption checks

| Assumption | Test | Result |
|---|---|---|
| Independence | By design | Satisfied, each run draws a fresh subset and trains from scratch |
| Normality | Shapiro-Wilk per group | Satisfied, all p > 0.05 (0.131, 0.581, 0.785, 0.245, 0.442) |
| Homogeneity of variance | Levene's test | **Violated**, W = 13.11, p = 1.08e-09 |

Normality was confirmed visually with Q-Q plots (`figures/qq_plots.png`).

### Test selection

Standard one-way ANOVA assumes equal variances across groups, which Levene's test rejected. Because the normality assumption held, a parametric approach was retained and **Welch's ANOVA** was used instead, which does not assume homogeneity of variance.

Rank-based (nonparametric) approaches were considered and rejected. They address non-normality, which was not the problem here, and converting accuracies to ranks discards the numeric magnitude of the differences.

**Post hoc:** Games-Howell, which is the unequal-variance counterpart to Tukey's HSD. Pairing Welch's ANOVA with Tukey would reintroduce the equal-variance assumption the correction exists to avoid.

---

## Results

### Descriptive statistics

| Training size | Mean accuracy | Std. deviation |
|---|---|---|
| 500 | 0.8500 | 0.006339 |
| 1000 | 0.8732 | 0.004468 |
| 2000 | 0.8863 | 0.004282 |
| 4000 | 0.8950 | 0.003069 |
| 8000 | 0.9036 | 0.002319 |

### Welch's ANOVA

```
F(4, 119.07) = 1099.28,  p = 5.85e-93,  partial eta squared = 0.951
```

The null hypothesis is rejected. Training set size has a statistically significant effect on classification accuracy, accounting for approximately 95% of the variance in accuracy.

### Games-Howell post hoc

All 10 pairwise comparisons are statistically significant (all p < 0.001). Hedges' g ranges from 2.32 (2000 vs 4000) to 11.14 (500 vs 8000). For reference, 0.8 is conventionally considered a large effect, so even the weakest comparison here is roughly three times that threshold.

### Key findings

1. **Accuracy increases with training set size.** Mean accuracy rises from 85.0% at n=500 to 90.4% at n=8000.

2. **Returns diminish with each doubling.** The gain from 500 to 1000 is 2.3 percentage points; from 4000 to 8000 it is 0.9 points. Each doubling of training data buys less than the previous one.

3. **Variance also decreases with training set size.** The standard deviation of accuracy falls from 0.0063 at n=500 to 0.0023 at n=8000. Larger training sets produce not only more accurate models but more consistent ones, because any two large random subsets resemble each other more closely than any two small ones. This is the direct cause of the unequal variances that Levene's test detected.

4. **Compute cost grows faster than accuracy.** Mean training time rises roughly 15-fold from the smallest to the largest condition, while accuracy gains only about 5 percentage points.

---

## Figures

| File | Description |
|---|---|
| `figures/qq_plots.png` | Q-Q plots per condition, normality assumption check |
| `figures/train_size_vs_accuracy.png` | Box plot of accuracy distributions by condition |
| `figures/mean_accuracy_trend.png` | Mean accuracy trend showing diminishing returns |
| `figures/pairwise_differences.png` | Games-Howell pairwise differences with error bars |
| `figures/accuracy_vs_time.png` | Accuracy against training time, the cost tradeoff |

---

## Repository Structure

```
mp2-statistical-analysis-mnist/
├── data/
│   ├── raw_results.csv              # 250 experimental runs
│   └── games_howell_results.csv     # pairwise comparison output
├── figures/                         # all generated figures
├── src/
│   ├── power_analysis.py            # determines replications per condition
│   ├── run_experiment.py            # runs the experiment, writes raw_results.csv
│   ├── analyze.py                   # assumption checks, Welch's ANOVA, Games-Howell
│   └── visualize.py                 # generates figures
├── requirements.txt
└── README.md
```

---

## Reproduction

### Setup

```bash
git clone https://github.com/Praful-Chunchu/mp2-statistical-analysis-mnist.git
cd mp2-statistical-analysis-mnist

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run

Scripts use paths relative to `src/`, so run them from that directory, in this order:

```bash
cd src

python power_analysis.py      # prints required replications per condition
python run_experiment.py      # writes ../data/raw_results.csv
python analyze.py             # prints all statistical results, writes games_howell_results.csv
python visualize.py           # writes figures to ../figures/
```


### Reproducibility

`run_experiment.py` sets `np.random.seed(42)` before drawing training subsets, so the results in this repository reproduce exactly. Removing the seed will produce slightly different accuracy values, though the pattern and all statistical conclusions are unaffected.

---

## Limitations

- The power analysis only compared two groups (500 vs 8000), not all five. 
  The 50 replications are sized for the biggest gap, so adjacent sizes like 
  4000 vs 8000 are less precisely covered.

- Assuming a medium effect (d = 0.5) turned out to be way too conservative. 
  The real effects were g = 2.32 to 11.14, so the study was overpowered. 
  Fewer runs would have reached the same conclusions.

- With 250 runs and effects this large, everything comes out significant. 
  Whether 0.86 percentage points is worth doubling the data from 4000 to 
  8000 is a separate question the p-value can't answer.

- This is MNIST and logistic regression only. A harder dataset or a bigger 
  model would likely give a different curve.

- Training sizes stopped at 8000 out of the 60,000 available, so the trend 
  shouldn't be extended past that.

- The same test set was used for every run. Good for consistency, but it 
  means all 250 accuracy numbers share whatever quirks that particular set has.

---

## Dependencies

See `requirements.txt`. Primary libraries:

- `scikit-learn` — logistic regression
- `keras` — MNIST dataset
- `pandas`, `numpy` — data handling
- `scipy` — Shapiro-Wilk, Levene's test
- `statsmodels` — power analysis, Q-Q plots
- `pingouin` — Welch's ANOVA, Games-Howell
- `matplotlib`, `seaborn` — figures