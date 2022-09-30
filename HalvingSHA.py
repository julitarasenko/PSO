from cmath import pi
import pandas as pd
import math
import numpy as np
import scipy.stats as ss
from pso import pso
from pso_domain import pso_domain 
import time

def HalvingSHA(generator_set, problem, dim, domain, max_iter):
    
    f = open('data.csv', 'a')

    #minimum resources
    r = 1

    # maximum resources
    R = 10
    base = 2

    sMax = math.floor(math.log(R/r, base))
    s = 0

    gen_len = len(generator_set)
    print(gen_len)

    n = len(generator_set['swarm']) * len(generator_set['omega'])*len(generator_set['phi_p'])*len(generator_set['phi_g'])
    print(n)

    # Generating table of setups from all combinations, where the list contains
    # 0 - swarm
    # 1 - omega
    setup = []
    for i1 in generator_set['swarm']:
        for i2 in generator_set['omega']:
            for i3 in generator_set['phi_p']:
                for i4 in generator_set['phi_g']:
                    setup.append([i1, i2, i3, i4])

    for i in range(0,(sMax-s)):
        ni = math.floor(n*math.pow(2, -i)) #number of setups for the iteration
        ri = r * math.pow(2, (i+s)) #number of resources in the iteration or swarm size?
        # print("Ri, Ni:", ri, ni)

        for j in range(1,ni):
            # run pso for given setup and given problem
            swarm_size = int(ri*10)
            # print("Swarm Size:", swarm_size)
            # print("Problem:", problem, dim, domain)
            # print("Setup:", setup[j])

            start = time.time()
            
            if len(domain) == np.size(domain):
                X, results = pso(dim, swarm_size, domain, setup[j], problem, max_iter)
            else:
                X, results = pso_domain(dim, swarm_size, domain, setup[j], problem, max_iter)

            finish = time.time()
            t = finish - start 
            gBest_fit = results[len(results) - 1][1]
            f.write(str(problem).partition(' ')[2].partition(' ')[0] + ';' + str(t) + ';' + str(gBest_fit))
            f.write('\n')   

    f.close()
    s
    return X, results

    # if gen_len == 3:
    #     return generator_set[0](generator_set[1], generator_set[2], size=10)
    # elif gen_len == 2:
    #     return generator_set[0](generator_set[1], size=10)
    # else:
    #     return generator_set[0](size=10)

# (ss.arcsine.rvs) = 0<x <1, dla omega and phi
# (ss.alpha.rvs) x>0, a>0 a=2 dla phi

