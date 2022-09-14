import scipy.stats as ss
from cmath import pi
from problem1 import problem1
from problem2 import problem2
from problem5 import problem5
from problem6 import problem6
from problem7 import problem7
from HalvingSHA import HalvingSHA

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
    'domain': [[-6.4, 6.5], [0, 4], [0, 4], [0, 4], [0, 2*pi]]
}

for i in range(5):
    problem = param_problem['name'][i]
    dim = param_problem['dim'][i]
    domain = param_problem['domain'][i]
    print(problem, dim, domain)
    HalvingSHA(random_gen, problem, dim, domain)