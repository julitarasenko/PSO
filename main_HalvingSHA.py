import numpy as np
import time
from joblib import Parallel, delayed
from HalvingSHA import HalvingSHA
from distribution_sets import distribution_sets
from test_sets import test_sets

generator_set = distribution_sets()
_, param_ftest = test_sets()
cores = 4
d_type = np.float128

def test(i):
    func = param_ftest['name'][i]
    dim = param_ftest['dim'][i]
    bound = param_ftest['domain'][i]
    min_value = param_ftest['min'][i]
    x_opt = param_ftest['x_best'][i]
    HalvingSHA(generator_set, func, dim, bound, min_value, x_opt, d_type)

start = time.time()
Parallel(n_jobs=cores)(delayed(test)(i) for i in range(np.size(param_ftest['name'])))
end = time.time()
print('{:.4f} s'.format(end-start))

# def problem(i):
#     problem = param_problem['name'][i]
#     dim = param_problem['dim'][i]
#     domain = param_problem['domain'][i]
#     exp_min = param_problem['min'][i]
#     HalvingSHA(generator_set, qmc_interval, problem, dim, domain, exp_min, False)

# start = time.time()
# Parallel(n_jobs=4)(delayed(problem)(i) for i in range(np.size(param_problem['name'])))
# end = time.time()
# print('{:.4f} s'.format(end-start))
