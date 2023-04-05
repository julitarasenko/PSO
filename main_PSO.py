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

def parallel_pso(j, domain, dim, setup, qmc_interval, problem, df_result):
    max_iter = 200 # might be connected with the algorithm as the resource
    swarm_size = 20 # might be connected with the algorithm as the resource

    start = time.time()
    
    if len(domain) == np.size(domain):
        results = pso(dim, swarm_size, domain, setup[j], j, qmc_interval, problem, max_iter)
    else:
        results = pso_domain(dim, swarm_size, domain, setup[j], j, qmc_interval, problem, max_iter)

    finish = time.time()
    t = (finish - start)

    if (j < qmc_interval[0] or j > qmc_interval[1]):
        decomposition_swarm = str(setup[j][0]).partition('_distns.')[2].partition(' object')[0]
    else:
        decomposition_swarm = str(setup[j][0]).partition('qmc.')[2].partition("'")[0]

    decomposition_w = str(setup[j][1]).partition('_distns.')[2].partition(' object')[0]
    decomposition_phi_p = str(setup[j][2]).partition('_distns.')[2].partition(' object')[0]
    decomposition_phi_g = str(setup[j][3]).partition('_distns.')[2].partition(' object')[0]
    
    df_new_row = pd.DataFrame([[j, decomposition_swarm, decomposition_w,
                                decomposition_phi_p, decomposition_phi_g,
                                swarm_size, max_iter]+ results + [t]], 
                                columns=df_result.columns)
    df_new_row.to_csv(f"resultPSO-{str(problem).split(' ')[1]}.csv", mode='a', header=False, index=False)  

def problem(i):
    problem = param_problem['name'][i]
    dim = param_problem['dim'][i]
    domain = param_problem['domain'][i]

    df_result = pd.DataFrame(columns=["setIdx", "Swarm", "Omega", "phiP", "phiG",
                                  "SwarmSize", "MaxIter", "BestFit", "avgSwarm",
                                  "stdSwarm",
                                  "MeanXCorr", "MeanVCorr", "Time"])
    
    df_result.to_csv(f"resultPSO-{str(problem).split(' ')[1]}.csv", index=False)

    Parallel(n_jobs=1)(delayed(parallel_pso)(j, domain, dim, setup, qmc_interval, problem, df_result) for j in range(n))
                                        
start = time.time()
Parallel(n_jobs=1)(delayed(problem)(i) for i in range(np.size(param_problem['name'])))
end = time.time()
print('{:.4f} s'.format(end-start))

def test(i):
    ftest = param_ftest['name'][i]
    dim = param_ftest['dim'][i]
    domain = param_ftest['domain'][i]

    df_result = pd.DataFrame(columns=["setIdx", "Swarm", "Omega", "phiP", "phiG",
                                  "SwarmSize", "MaxIter", "BestFit", "avgSwarm",
                                  "stdSwarm",
                                  "MeanXCorr", "MeanVCorr", "Time"])
    
    df_result.to_csv(f"resultPSO-{str(problem).split(' ')[1]}.csv", index=False)
    
    Parallel(n_jobs=1)(delayed(parallel_pso)(j, domain, dim, setup, qmc_interval, ftest, df_result) for j in range(n))

start = time.time()
Parallel(n_jobs=2)(delayed(test)(i) for i in range(np.size(param_ftest['name'])))
end = time.time()
print('{:.4f} s'.format(end-start))



