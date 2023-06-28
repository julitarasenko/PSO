import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(123)

# Generate 20 random variables from the Cauchy distribution
cauchy = np.random.standard_cauchy((20, 2))

# Generate 20 random variables from the Student's t-distribution
t = np.random.standard_t(3, (20, 2))

# Generate 20 random variables from the Pareto-Levy distribution
pareto_levy = np.random.pareto(1.5, (20, 2)) * np.random.standard_cauchy((20, 2))

# Generate 20 random variables from the Weibull distribution
weibull = np.random.weibull(2, (20, 2))

# Generate 20 random variables from the Log-Normal distribution
lognormal = np.random.lognormal(0, 1, (20, 2))

# Generate 20 random variables from the Levy distribution
# levy = np.random.levy(0.5, (20, 2))

# Create a figure with subplots for each distribution
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

# Set titles for each subplot
titles = ['Cauchy Distribution', "Student's t-Distribution", 'Pareto-Levy Distribution', 'Weibull Distribution', 'Log-Normal Distribution']

# Plot each distribution on a separate subplot
for ax, data, title in zip(axes.flatten(), [cauchy, t, pareto_levy, weibull, lognormal], titles):
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