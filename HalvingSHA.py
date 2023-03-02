import pandas as pd
import math
import numpy as np
from pso import pso
from pso_domain import pso_domain 
import time
from joblib import Parallel, delayed

def HalvingSHA(generator_set, qmc_interval, problem, dim, domain, ex_min):

    #minimum resources
    r = 2

    # maximum resources
    R = len(generator_set['swarm']) * len(generator_set['omega']) * len(generator_set['phi_p']) * len(generator_set['phi_g'])
    base = 2

    sMax = math.ceil(math.log(R/r, base))
    s = 0


    # Generating table of setups from all combinations, where the list contains
    # 0 - swarm
    # 1 - omega
    # 2 - phi_p
    # 3 - phi_g
    setup = [] # store all combinations of 4 params combinations
    for i1 in generator_set['swarm']:
        for i2 in generator_set['omega']:
            for i3 in generator_set['phi_p']:
                for i4 in generator_set['phi_g']:
                    setup.append([i1, i2, i3, i4])

    df_result = pd.DataFrame(columns=["setIdx", "Swarm", "Omega", "phiP", "phiG",
                                          "SwarSize", "MaxIter", "Iteration", 
                                          "BestFit", "avgSwarm", "stdSwarm",
                                          "MeanXCorr", "MeanVCorr", "Time"])
    df_result.to_csv(f"result-{str(problem).split(' ')[1]}.csv", index=False)

    # The halving algorithm
    index = list(range(R))
    for i in range(0,(sMax-s)):
        ni = math.floor(R*math.pow(2, -i)) #number of setups for the iteration
        ri = r * math.pow(2, (i+s)) #number of resources in the iteration or swarm size?

        Parallel(n_jobs=1)(delayed(parallel_pso)(j, ri, domain, dim, setup, qmc_interval, problem, df_result) for j in index[:ni])
        
        df_result["Criteria"] = df_result['BestFit']-ex_min + df_result['MeanXCorr'] + df_result['MeanVCorr']
        df_result['BestRank'] = df_result['Criteria'].rank(ascending=False, pct=True)
        df_result.to_csv(f"result-{str(problem).split(' ')[1]}.csv", mode='a', header=False, index=False)
        # Sort by rank and store in index vector
        df_result.sort_values(by='BestRank', ascending=False, inplace=True)
        index = df_result['setIdx'].values
        df_result = pd.DataFrame(columns=["setIdx", "Swarm", "Omega", "phiP", "phiG",
                                          "SwarSize", "MaxIter", "Iteration", 
                                          "BestFit", "avgSwarm", "stdSwarm",
                                          "MeanXCorr", "MeanVCorr", "Time"])


def parallel_pso(j, ri, domain, d, setup, qmc_interval, problem, df_result):
    max_iter = int(ri*2) # might be connected with the algorithm as the resource
    swarm_size = int(ri*2) # might be connected with the algorithm as the resource
    iteration = int(ri*2)

    sets = setup[j]

    if len(sets[1]) == 1:
        w = sets[1][0](size=1)
    else:
        w = sets[1][0](sets[1][1],size=1)
    # Personal influence factor
    if len(sets[2]) == 1:
        phi_p = sets[2][0](size=d)
    else:
        phi_p = sets[2][0](sets[2][1],size=d)
    # Global influence factor
    if len(sets[3]) == 1:
        phi_g = sets[3][0](size=d)
    else:
        phi_g = sets[3][0](sets[3][1], size=d)

    # # Create particles inside the range
    if (len(sets[0]) == 1 and (j < qmc_interval[0] or j > qmc_interval[1])):
        X_ = sets[0][0](size=(swarm_size, d))
    else:
        X_ = sets[0][0](d=d).random(n=swarm_size)

    start = time.time()
    
    results = []
    for i in range(iteration):
        if len(domain) == np.size(domain):
            result = pso(d, swarm_size, domain, w, phi_p, phi_g, X_, problem, max_iter)
        else:
            result = pso_domain(d, swarm_size, domain, w, phi_p, phi_g, X_, problem, max_iter)
        results.append(result)

    finish = time.time()
    t = (finish - start)/iteration

    result_mean = list(np.mean(results, axis=0))

    if (j < qmc_interval[0] or j > qmc_interval[1]):
        df_result.loc[len(df_result)] = [j, str(setup[j][0]).partition('_distns.')[2].partition(' object')[0],
                    str(setup[j][1]).partition('_distns.')[2].partition(' object')[0],
                    str(setup[j][2]).partition('_distns.')[2].partition(' object')[0],
                    str(setup[j][3]).partition('_distns.')[2].partition(' object')[0],
                    swarm_size, max_iter, iteration]+ result_mean + [t]
    else:
        df_result.loc[len(df_result)] = [j, str(setup[j][0]).partition('qmc.')[2].partition("'")[0],
                    str(setup[j][1]).partition('_distns.')[2].partition(' object')[0],
                    str(setup[j][2]).partition('_distns.')[2].partition(' object')[0],
                    str(setup[j][3]).partition('_distns.')[2].partition(' object')[0],
                    swarm_size, max_iter, iteration]+ result_mean + [t]


