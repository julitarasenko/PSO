import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy, t, pareto, weibull_min, lognorm, levy_stable

# Set random seed for reproducibility
np.random.seed(123)

# Generate 20 random variables from the Cauchy distribution
cauchy_rvs = cauchy.rvs(size=(20, 2))

# Generate 20 random variables from the Student's t-distribution
t_rvs = t.rvs(df=3, size=(20, 2))

# Generate 20 random variables from the Pareto-Levy distribution
pareto_rvs = pareto.rvs(b=1.5, size=(20, 2)) * cauchy.rvs(size=(20, 2))

# Generate 20 random variables from the Weibull distribution
weibull_rvs = weibull_min.rvs(c=2, size=(20, 2))

# Generate 20 random variables from the Log-Normal distribution
lognormal_rvs = lognorm.rvs(s=1, size=(20, 2))

# Generate 20 random variables from the Levy distribution
levy_rvs = levy_stable.rvs(alpha=0.5, beta=0, loc=0, scale=1, size=(20, 2))

# Create a figure with subplots for each distribution
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

# Set titles for each subplot
titles = ['Cauchy Distribution', "Student's t-Distribution", 'Pareto-Levy Distribution', 'Weibull Distribution', 'Log-Normal Distribution', 'Levy Distribution']

# Plot each distribution on a separate subplot
for ax, data, title in zip(axes.flatten(), [cauchy_rvs, t_rvs, pareto_rvs, weibull_rvs, lognormal_rvs, levy_rvs], titles):
    ax.scatter(data[:, 0], data[:, 1], alpha=0.8, edgecolors='none')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Add a title for the entire figure
fig.suptitle('Random Variables from Different Distributions', fontsize=16)

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()