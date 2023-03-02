import math
import numpy as np
import time
from joblib import Parallel, delayed
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

generator_set = {  
            'swarm': [(ss.norm.rvs,), 
                        (ss.uniform.rvs,), 
                        (ss.qmc.Sobol,), #start library scipy.stats.qmc in index 2
                        (ss.qmc.Halton,), 
                        (ss.qmc.LatinHypercube,), #end ibrary scipy.stats.qmc in index 4
                        # (ss.qmc.PoissonDisk(d=dim),), 
                        (ss.levy.rvs,)],

            'omega': [(ss.arcsine.rvs,),
                        (ss.norm.rvs,), 
                        (ss.uniform.rvs,)],

            'phi_p': [(ss.alpha.rvs, 2), 
                        (ss.arcsine.rvs,), 
                        (ss.uniform.rvs,)],

            'phi_g': [(ss.arcsine.rvs,), 
                        (ss.uniform.rvs,)]
        }

qmc_start_index = 2
qmc_end_index = 4

qmc_interval = [len(generator_set['omega']) * len(generator_set['phi_p']) * len(generator_set['phi_g'])*qmc_start_index,
                len(generator_set['omega']) * len(generator_set['phi_p']) * len(generator_set['phi_g'])*(qmc_end_index+1)-1]

n=5
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

def problem(i):
    problem = param_problem['name'][i]
    dim = param_problem['dim'][i]
    domain = param_problem['domain'][i]
    exp_min = param_problem['min'][i]
    HalvingSHA(generator_set, qmc_interval, problem, dim, domain, exp_min)

start = time.time()
Parallel(n_jobs=1)(delayed(problem)(i) for i in range(np.size(param_problem['name'])))
end = time.time()
print('{:.4f} s'.format(end-start))

n = 20

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

def test(i):
    ftest = param_ftest['name'][i]
    dim = param_ftest['dim'][i]
    domain = param_ftest['domain'][i]
    exp_min = param_ftest['min'][i]
    HalvingSHA(generator_set, qmc_interval, ftest, dim, domain, exp_min)

start = time.time()
Parallel(n_jobs=2)(delayed(test)(i) for i in range(np.size(param_ftest['name'])))
end = time.time()
print('{:.4f} s'.format(end-start))

