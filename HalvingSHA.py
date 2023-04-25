from cmath import pi
import pandas as pd
import math
import numpy as np
import scipy.stats as ss
from pso import pso
from pso_domain import pso_domain 
import time
from joblib import Parallel, delayed

def HalvingSHA(generator_set, qmc_interval, problem, dim, domain, exp_min, test):

    #minimum resources
    r = 100

    # maximum resources
    R = len(generator_set['swarm']) * len(generator_set['omega']) * len(generator_set['phi_p']) * len(generator_set['phi_g'])

    # print("R: ", R)
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
                                      "SwarmSize", "MaxIter", "BestFit", "avgSwarm",
                                      "stdSwarm",
                                      "MeanXCorr", "MeanVCorr", "Iter", "Time",])
    df_result.to_csv(f"resultSHA-{str(problem).split(' ')[1]}.csv", index=False)

    # The halving algorithm
    index = list(range(R))
    for i in range(0,(sMax-s)):
        ni = math.floor(R*math.pow(2, -i)) #number of setups for the iteration
        ri = r * math.pow(2, (i+s)) #number of resources in the iteration or swarm size?

        Parallel(n_jobs=1)(delayed(parallel_pso)(j, ri, domain, dim, setup, qmc_interval, problem, df_result, exp_min, test) for j in index[:ni])
        
        df_result["Criteria"] = df_result['BestFit'] - exp_min + df_result['MeanXCorr'] + df_result['MeanVCorr']
        df_result['BestRank'] = df_result['Criteria'].rank(ascending=False, pct=True)
        # Sort by rank and store in index vector
        df_result.sort_values(by='BestRank', ascending=False, inplace=True)
        index = df_result['setIdx'].values
        df_result = pd.DataFrame(columns=["setIdx", "Swarm", "Omega", "phiP", "phiG",
                                          "SwarmSize", "MaxIter", "BestFit", "avgSwarm",
                                          "stdSwarm",
                                          "MeanXCorr", "MeanVCorr", "Iter" "Time"])                                          

def parallel_pso(j, ri, domain, dim, setup, qmc_interval, problem, df_result, exp_min, test):
    max_iter = int(ri / 25) # might be connected with the algorithm as the resource
    swarm_size = 10 # might be connected with the algorithm as the resource

    start = time.time()
    
    if len(domain) == np.size(domain):
        results = pso(dim, swarm_size, domain, setup[j], j, qmc_interval, problem, max_iter, exp_min, test)
    else:
        results = pso_domain(dim, swarm_size, domain, setup[j], j, qmc_interval, problem, max_iter, exp_min, test)

    finish = time.time()
    t = round(finish - start, 5)

    if (j < qmc_interval[0] or j > qmc_interval[1]):
        decomposition_swarm = str(setup[j][0]).partition('_distns.')[2].partition(' object')[0]
    else:
        decomposition_swarm = str(setup[j][0]).partition('qmc.')[2].partition("'")[0]

    decomposition_w = str(setup[j][1]).partition('_distns.')[2].partition(' object')[0]
    decomposition_phi_p = str(setup[j][2]).partition('_distns.')[2].partition(' object')[0]
    decomposition_phi_g = str(setup[j][3]).partition('_distns.')[2].partition(' object')[0]

    df_result.loc[len(df_result)] = [j, decomposition_swarm, decomposition_w,
                                     decomposition_phi_p, decomposition_phi_g,
                                     swarm_size, max_iter]+ results + [t]
    df_new_row = pd.DataFrame([[j, decomposition_swarm, decomposition_w,
                                decomposition_phi_p, decomposition_phi_g,
                                swarm_size, max_iter]+ results + [t]], 
                                columns=df_result.columns)
    df_new_row.to_csv(f"resultSHA-{str(problem).split(' ')[1]}.csv", mode='a', header=False, index=False)  
    
    


