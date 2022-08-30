import pandas as pd
import math
import scipy.stats as ss
from pso import pso

def HalvingSHA( generator_set):
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
                    #print(setup)

    for i in range(0,(sMax-s)):
        ni = math.floor(n*math.pow(2, -i)) #number of setups for the iteration
        ri = r * math.pow(2, (i+s)) #number of resources in the iteration

        for j in range(1,ni):
            pso(3,10,[-10,10],setup)

    # if gen_len == 3:
    #     return generator_set[0](generator_set[1], generator_set[2], size=10)
    # elif gen_len == 2:
    #     return generator_set[0](generator_set[1], size=10)
    # else:
    #     return generator_set[0](size=10)

# (ss.arcsine.rvs) = 0<x <1, dla omega and phi
# (ss.alpha.rvs) x>0, a>0 a=2 dla phi
random_gen = {'swarm': [(ss.norm.rvs,), (ss.uniform.rvs,)],
        # (ss.f.rvs, 29, 18), (ss.levy.rvs, 10, 2),],
        'omega': [(ss.arcsine.rvs),(ss.norm.rvs,), (ss.uniform.rvs,)],
        # (ss.f.rvs, 29, 18), (ss.levy.rvs, 10, 2),],
        'phi_p': [(ss.alpha.rvs, 2), (ss.arcsine.rvs),
                (ss.uniform.rvs,)],
        'phi_g': [(ss.alpha.rvs,2),(ss.arcsine.rvs), (ss.uniform.rvs,)]
        }
print(HalvingSHA( random_gen))