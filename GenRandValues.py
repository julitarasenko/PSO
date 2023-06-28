from distribution_sets import distribution_sets
from scipy import stats as ss
from itertools import combinations

generator_set, qmc_interval = distribution_sets()
# print(generator_set)
# print(qmc_interval)
loc, scale = 0, 1

n = len(generator_set['swarm']) * len(generator_set['phi_p']) * len(generator_set['phi_g'])
print(n)

# # generate samples from two distributions
# size_points = 200
# for i in generator_set['phi_p']:
#     print(f"Rozkład: {i.dist.name}")
#     for j in generator_set['phi_g']:
#         if i != j:
#             sample1 = i.rvs(size=size_points)
#             sample2 = j.rvs(size=size_points)
#
#             # perform Kolmogoro v-Smirnov test
#             ks_stat, p_value = ss.ks_2samp(sample1, sample2)
#
#             if ks_stat > 0.2:
#                 print(f"{j.dist.name} ... K-S statistic: {ks_stat}... P-value: {p_value}")

def test_significant_difference(distribution1, distribution2):
    sample_size = 200
    sample1 = distribution1.rvs(size=sample_size)
    sample2 = distribution2.rvs(size=sample_size)
    ks_stat, p_value = ss.ks_2samp(sample1, sample2)
    return ks_stat, p_value

significant_pairs = []
alpha = 0.05  # Significance level

# Generate all pairs of distributions
pair_combinations = combinations(generator_set['omega'], 2)
print(pair_combinations)

for dist1, dist2 in pair_combinations:
    ks_stat, p_value = test_significant_difference(dist1, dist2)
    if p_value < alpha:
        significant_pairs.append((dist1, dist2, ks_stat, p_value))


# Print significant pairs
for dist1, dist2, ks_stat, p_value in significant_pairs:
    print(f"Distributions {dist1.dist.name} and {dist2.dist.name} are significantly different.")
    print(f"K-S statistic: {ks_stat}")
    print(f"P-value: {p_value}")
    print()