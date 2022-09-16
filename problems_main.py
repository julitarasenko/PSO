import numpy as np
from scipy import stats as ss
from cmath import inf, pi
from problem1 import problem1
from problem2 import problem2
from problem5 import problem5
from problem6 import problem6
from problem7 import problem7
from HalvingSHA import HalvingSHA
from plots import spheref, zakharov, rosenbrock, modified_rosenbrock, easom, ackley, griewank, alpine, perm, schwefel, yang3, yang4, csendes, yang2, levy

random_gen = {'swarm': [(ss.norm.rvs,), (ss.uniform.rvs,)],
        # (ss.f.rvs, 29, 18), (ss.levy.rvs, 10, 2),],
        'omega': [(ss.arcsine.rvs,),(ss.norm.rvs,), (ss.uniform.rvs,)],
        # (ss.f.rvs, 29, 18), (ss.levy.rvs, 10, 2),],
        'phi_p': [(ss.alpha.rvs, 2), (ss.arcsine.rvs,), (ss.uniform.rvs,)],
        'phi_g': [(ss.arcsine.rvs,), (ss.uniform.rvs,)]
        }

param_problem = {
    'name' : [problem1, problem2, problem5, problem6, problem7],
    'dim': [6, 3, 3, 3, 20],
    'domain': [[-6.4, 6.5], [[0, 4], [0, 4], [0, np.pi]], [[0, 4], [0, 4], [0, np.pi]], [[0, 4], [0, 4], [0, np.pi]], [0, 2*pi]]
}

# param_problem = {
#     'name' : [problem2],
#     'dim': [3],
#     'domain': [[[0, 4], [0, 4], [0, np.pi]]]
# }

for i in range(np.size(param_problem['name'])):
    problem = param_problem['name'][i]
    dim = param_problem['dim'][i]
    domain = param_problem['domain'][i]
    print(problem, dim, domain)
    HalvingSHA(random_gen, problem, dim, domain)


param_ftest = {
    'name' : [spheref, zakharov, rosenbrock, modified_rosenbrock, easom, ackley, griewank, alpine, perm, schwefel, yang3, yang4, csendes, yang2, levy],
    'dim' : [inf, inf, inf, 4, 2, inf, inf, inf, inf, inf, inf, inf, inf, inf, inf],
    'domain' : [[0, 10], [-5, 5], [-5, 10], [0, 1], [-100, 100], [-35, 35], [-100, 100], [-10, 10], [-dim, dim], [-500, 500], [-2*pi, 2*pi], [-20, 20], [-1, 1], [-5, 5], [-10, 10]],
    'min' : [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0]
}

for i in range(np.size(param_ftest['name'])):
    ftest = param_ftest['name'][i]
    dim = param_ftest['dim'][i]
    if dim == inf:
        dim = 10
    domain = param_ftest['domain'][i]
    print(ftest, dim, domain)
    HalvingSHA(random_gen, ftest, dim, domain)