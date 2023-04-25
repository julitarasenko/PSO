import numpy as np
import time
from joblib import Parallel, delayed
import pandas as pd
from pso import pso
from pso_domain import pso_domain
from distribution_sets import distribution_sets
from test_sets import test_sets

generator_set, qmc_interval = distribution_sets()
param_problem, param_ftest = test_sets()

setup = [] # store all combinations of 4 params combinations
for i1 in generator_set['swarm']:
    for i2 in generator_set['omega']:
        for i3 in generator_set['phi_p']:
            for i4 in generator_set['phi_g']:
                setup.append([i1, i2, i3, i4])

n = len(generator_set['swarm']) * len(generator_set['omega']) * len(generator_set['phi_p']) * len(generator_set['phi_g'])

def parallel_pso(j, domain, dim, setup, qmc_interval, problem, df_result, exp_min, test):
    max_iter = 20000 # might be connected with the algorithm as the resource
    swarm_size = 10 # might be connected with the algorithm as the resource

    start = time.time()
    
    if len(domain) == np.size(domain):
        results = pso(dim, swarm_size, domain, setup[j], j, qmc_interval, problem, max_iter, exp_min, test)
    else:
        results = pso_domain(dim, swarm_size, domain, setup[j], j, qmc_interval, problem, max_iter, exp_min, test)

    finish = time.time()
    t = round(finish - start, 5)

    if (j < qmc_interval[0] or j > qmc_interval[1]):
        decomposition_swarm = setup[j][0].dist.name
    else:
        decomposition_swarm = str(setup[j][0]).partition('qmc.')[2].partition("'")[0]
        
    decomposition_w = setup[j][1].dist.name
    decomposition_phi_p = setup[j][2].dist.name
    decomposition_phi_g = setup[j][3].dist.name
    
    df_new_row = pd.DataFrame([[j, decomposition_swarm, decomposition_w,
                                decomposition_phi_p, decomposition_phi_g,
                                swarm_size, max_iter]+ results + [t]], 
                                columns=df_result.columns)
    df_new_row.to_csv(f"resultPSO-{str(problem).split(' ')[1]}.csv", mode='a', header=False, index=False)  

def test(i):
    ftest = param_ftest['name'][i]
    dim = param_ftest['dim'][i]
    domain = param_ftest['domain'][i]
    exp_min = param_problem['min'][i]

    df_result = pd.DataFrame(columns=["setIdx", "Swarm", "Omega", "phiP", "phiG",
                                  "SwarmSize", "MaxIter", "BestFit", "avgSwarm",
                                  "stdSwarm",
                                  "MeanXCorr", "MeanVCorr", "Iter", "Time"])
    
    df_result.to_csv(f"resultPSO-{str(ftest).split(' ')[1]}.csv", index=False)
    
    Parallel(n_jobs=1)(delayed(parallel_pso)(j, domain, dim, setup, qmc_interval, ftest, df_result, exp_min, True) for j in range(n))

start = time.time()
Parallel(n_jobs=1)(delayed(test)(i) for i in range(np.size(param_ftest['name'])))
end = time.time()
print('{:.4f} s'.format(end-start))


def problem(i):
    problem = param_problem['name'][i]
    dim = param_problem['dim'][i]
    domain = param_problem['domain'][i]
    exp_min = param_problem['min'][i] 

    df_result = pd.DataFrame(columns=["setIdx", "Swarm", "Omega", "phiP", "phiG",
                                  "SwarmSize", "MaxIter", "BestFit", "avgSwarm",
                                  "stdSwarm",
                                  "MeanXCorr", "MeanVCorr", "Iter", "Time"])
    
    df_result.to_csv(f"resultPSO-{str(problem).split(' ')[1]}.csv", index=False)

    Parallel(n_jobs=2)(delayed(parallel_pso)(j, domain, dim, setup, qmc_interval, problem, df_result, exp_min, False) for j in range(n))
                                        
start = time.time()
Parallel(n_jobs=2)(delayed(problem)(i) for i in range(np.size(param_problem['name'])))
end = time.time()
print('{:.4f} s'.format(end-start))








