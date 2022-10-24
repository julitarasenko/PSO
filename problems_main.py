import math

import numpy as np
from scipy import stats as ss
from cmath import inf, pi
from problem1 import problem1
from problem2 import problem2
from problem5 import problem5
from problem6 import problem6
from problem7 import problem7
from HalvingSHA import HalvingSHA
from plots import spheref, zakharov, rosenbrock, trid6, easom, ackley, griewank, alpine, perm, schwefel, \
    yang3, yang4, csendes, yang2, levy8


n=5
random_gen = {'swarm': [(ss.norm.rvs,), (ss.uniform.rvs,)],
        # (ss.f.rvs, 29, 18), (ss.levy.rvs, 10, 2),],
        'omega': [(ss.arcsine.rvs,),(ss.norm.rvs,), (ss.uniform.rvs,)],
        # (ss.f.rvs, 29, 18), (ss.levy.rvs, 10, 2),],
        'phi_p': [(ss.alpha.rvs, 2), (ss.arcsine.rvs,), (ss.uniform.rvs,)],
        'phi_g': [(ss.arcsine.rvs,), (ss.uniform.rvs,)]
        }

param_problem = {
    'name' : [problem1, problem2, problem5, problem6, problem7],
    'dim': [20, 6, 3*n, 3*n, 3*n, 20],
    'domain': [[-6.4, 6.5], # problem1
               [[0, 4], [0, 4], [0, np.pi]] + [[-4-1/4*math.floor((i-4)/3), +4+1/4*math.floor((i-4)/3)] for i in range(4,3*n+1)],
               # [[-4.5, 4.25], [0, 4], [0, np.pi]] + [[-4.25, 4.25] for i in range(4,3*n+1)], # problem5
               # [[-4.5, 4.25], [0, 4], [0, np.pi]] + [[-4.25, 4.25] for i in range(4,3*n+1)], # problem6
               [[0, 4], [0, 4], [0, np.pi]] + [
                   [-4 - 1 / 4 * math.floor((i - 4) / 3), +4 + 1 / 4 * math.floor((i - 4) / 3)] for i in
                   range(4, 3 * n + 1)],
               [[0, 4], [0, 4], [0, np.pi]] + [
                   [-4 - 1 / 4 * math.floor((i - 4) / 3), +4 + 1 / 4 * math.floor((i - 4) / 3)] for i in
                   range(4, 3 * n + 1)],
               [0, 2*pi]],
    'min': [0, -600, -600, -600, 0]
}

n = 20
for i in range(np.size(param_problem['name'])):
    problem = param_problem['name'][i]
    dim = param_problem['dim'][i]
    domain = param_problem['domain'][i]
    print(problem, dim, domain)
    exp_min = param_problem['min'][i]
    print(problem, dim, domain)
    HalvingSHA(random_gen, problem, dim, domain, exp_min)

    

param_ftest = {
    'name': [spheref, zakharov, rosenbrock, trid6, easom, ackley, griewank, alpine, perm, schwefel, yang3,
             yang4, csendes, yang2, levy8],
    'dim': [n, n, n, n, 2, n, n, n, n, n, n, n, n, n, n],
    'domain': [[0, 10], [-5, 5], [-5, 10], # spheref, zakharov, rosenbrock,
                [-36, 36], [-100, 100], [-35, 35], # trid6, easom, ackley,
                [-100, 100], [-10, 10], [-n, n], # griewank, alpine, perm,
               [-500, 500], [-2*pi, 2*pi], [-20, 20], # schwefel, yang3, yang4,
               [-1, 1], [-5, 5], [-10, 10]], # csendes, yang2, levy
    'min': [0, 0, 0, -50, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0]
}

for i in range(np.size(param_ftest['name'])):
    ftest = param_ftest['name'][i]
    dim = param_ftest['dim'][i]
    domain = param_ftest['domain'][i]
    exp_min = param_ftest['min'][i]
    print(ftest, dim, domain)
    HalvingSHA(random_gen, ftest, dim, domain, exp_min)

