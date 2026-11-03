import math
from statsmodels.stats.power import TTestIndPower
"""Determines the required number of replications per condition via power analysis."""
# Parameters
effect_size = 0.5 #Cohen's d - we want a medium effect
alpha = 0.05
power = 0.80

# Power Analysis
analysis = TTestIndPower()
n_size = analysis.solve_power(
    effect_size = effect_size,
    alpha = alpha,
    power = power,
    alternative = 'larger'
)


print(n_size)