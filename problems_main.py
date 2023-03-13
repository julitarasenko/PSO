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

loc, scale = 0, 1
generator_set = {  
            'swarm': [(ss.norm.rvs,), 
                        (ss.uniform.rvs,), 
                        (ss.qmc.Sobol,), #start library scipy.stats.qmc in index 2
                        (ss.qmc.Halton,), 
                        (ss.qmc.LatinHypercube,), #end ibrary scipy.stats.qmc in index 4
                        # (ss.qmc.PoissonDisk,), 
                        (ss.levy.rvs,)],

            'omega': [(ss.alpha.rvs, 3.57,),
                        (ss.arcsine.rvs,), 
                        (ss.argus.rvs, 1,),
                        (ss.beta.rvs, 2.31, 0.63,),
                        (ss.betaprime.rvs, 5, 6,),
                        (ss.bradford.rvs, 0.3,),
                        (ss.burr.rvs, 10.5, 4.3,),
                        (ss.burr12.rvs, 10, 4,),
                        (ss.chi.rvs, 78,),
                        (ss.chi2.rvs, 55,),
                        (ss.erlang.rvs, 10,),
                        (ss.expon.rvs,),
                        (ss.exponpow.rvs, 2.7,),
                        (ss.exponweib.rvs, 2.89, 1.95,),
                        (ss.f.rvs, 29, 18,),
                        (ss.fatiguelife.rvs, 29,),
                        (ss.fisk.rvs, 3.09,),
                        (ss.foldcauchy.rvs, 4.72,),
                        (ss.foldnorm.rvs, 1.95,),
                        (ss.gamma.rvs, 1.99,),
                        (ss.gausshyper.rvs, 13.76, 3.12, 2.51, 5.18,),
                        (ss.genexpon.rvs, 9.13, 16.23, 3.28,),
                        (ss.gengamma.rvs, 4.42, 3.12,),
                        (ss.gengamma.rvs, 4.42, -3.12,),
                        (ss.genhalflogistic.rvs, 0.77,),
                        (ss.geninvgauss.rvs, 2.3, 1.5,),
                        (ss.halfgennorm.rvs, 0.67,),
                        (ss.genpareto.rvs, 0.1,),
                        (ss.gilbrat.rvs,),
                        (ss.gompertz.rvs, 0.95,),
                        (ss.halfcauchy.rvs,),
                        (ss.halflogistic.rvs,),
                        (ss.halfnorm.rvs,),
                        (ss.invgamma.rvs, 4.07,),
                        (ss.invgauss.rvs, 0.15,),
                        (ss.invweibull.rvs, 10.58,),
                        (ss.johnsonsb.rvs, 4.32, 3.18,),
                        (ss.kappa3.rvs, 1, ),
                        (ss.ksone.rvs, 1000,),
                        (ss.kstwo.rvs, 10,),
                        (ss.kstwobign.rvs,),
                        (ss.levy.rvs,),
                        (ss.loglaplace.rvs, 3.25,),
                        (ss.lognorm.rvs, 0.95,),
                        (ss.loguniform.rvs, 0.01, 1,),
                        (ss.lomax.rvs, 1.88,),
                        (ss.maxwell.rvs,),
                        (ss.mielke.rvs, 10.4, 4.6,),
                        (ss.nakagami.rvs, 4.97,),
                        (ss.ncf.rvs, 27, 27, 0.42,),
                        (ss.ncx2.rvs, 21, 1.06,),
                        (ss.pareto.rvs, 2.62,),
                        (ss.powerlaw.rvs, 1.66,),
                        (ss.powerlognorm.rvs, 2.14, 0.45,),
                        (ss.rayleigh.rvs,),
                        (ss.recipinvgauss.rvs, 0.63,),
                        (ss.reciprocal.rvs, 0.01, 1,),
                        (ss.rice.rvs, 0.77,),
                        (ss.trapz.rvs, 0.2, 0.8,),
                        (ss.triang.rvs, 0.16,),
                        (ss.truncexpon.rvs, 4.69,),
                        (ss.truncnorm.rvs, 0.1, 2,),
                        (ss.uniform.rvs,),
                        (ss.wald.rvs,),
                        (ss.weibull_min.rvs, 1.79,),
                        (ss.wrapcauchy.rvs, 0.03,)],

            'phi_p': [(ss.alpha.rvs, 3.57,),
                        (ss.arcsine.rvs,), 
                        (ss.argus.rvs, 1,),
                        (ss.beta.rvs, 2.31, 0.63,),
                        (ss.betaprime.rvs, 5, 6,),
                        (ss.bradford.rvs, 0.3,),
                        (ss.burr.rvs, 10.5, 4.3,),
                        (ss.burr12.rvs, 10, 4,),
                        (ss.chi.rvs, 78,),
                        (ss.chi2.rvs, 55,),
                        (ss.erlang.rvs, 10,),
                        (ss.expon.rvs,),
                        (ss.exponpow.rvs, 2.7,),
                        (ss.exponweib.rvs, 2.89, 1.95,),
                        (ss.f.rvs, 29, 18,),
                        (ss.fatiguelife.rvs, 29,),
                        (ss.fisk.rvs, 3.09,),
                        (ss.foldcauchy.rvs, 4.72,),
                        (ss.foldnorm.rvs, 1.95,),
                        (ss.gamma.rvs, 1.99,),
                        (ss.gausshyper.rvs, 13.76, 3.12, 2.51, 5.18,),
                        (ss.genexpon.rvs, 9.13, 16.23, 3.28,),
                        (ss.gengamma.rvs, 4.42, 3.12,),
                        (ss.gengamma.rvs, 4.42, -3.12,),
                        (ss.genhalflogistic.rvs, 0.77,),
                        (ss.geninvgauss.rvs, 2.3, 1.5,),
                        (ss.halfgennorm.rvs, 0.67,),
                        (ss.genpareto.rvs, 0.1,),
                        (ss.gilbrat.rvs,),
                        (ss.gompertz.rvs, 0.95,),
                        (ss.halfcauchy.rvs,),
                        (ss.halflogistic.rvs,),
                        (ss.halfnorm.rvs,),
                        (ss.invgamma.rvs, 4.07,),
                        (ss.invgauss.rvs, 0.15,),
                        (ss.invweibull.rvs, 10.58,),
                        (ss.johnsonsb.rvs, 4.32, 3.18,),
                        (ss.kappa3.rvs, 1, ),
                        (ss.ksone.rvs, 1000,),
                        (ss.kstwo.rvs, 10,),
                        (ss.kstwobign.rvs,),
                        (ss.levy.rvs,),
                        (ss.loglaplace.rvs, 3.25,),
                        (ss.lognorm.rvs, 0.95,),
                        (ss.loguniform.rvs, 0.01, 1,),
                        (ss.lomax.rvs, 1.88,),
                        (ss.maxwell.rvs,),
                        (ss.mielke.rvs, 10.4, 4.6,),
                        (ss.nakagami.rvs, 4.97,),
                        (ss.ncf.rvs, 27, 27, 0.42,),
                        (ss.ncx2.rvs, 21, 1.06,),
                        (ss.pareto.rvs, 2.62,),
                        (ss.powerlaw.rvs, 1.66,),
                        (ss.powerlognorm.rvs, 2.14, 0.45,),
                        (ss.rayleigh.rvs,),
                        (ss.recipinvgauss.rvs, 0.63,),
                        (ss.reciprocal.rvs, 0.01, 1,),
                        (ss.rice.rvs, 0.77,),
                        (ss.trapz.rvs, 0.2, 0.8,),
                        (ss.triang.rvs, 0.16,),
                        (ss.truncexpon.rvs, 4.69,),
                        (ss.truncnorm.rvs, 0.1, 2,),
                        (ss.uniform.rvs,),
                        (ss.wald.rvs,),
                        (ss.weibull_min.rvs, 1.79,),
                        (ss.wrapcauchy.rvs, 0.03,)],

            'phi_g': [(ss.alpha.rvs, 3.57,),
                        (ss.arcsine.rvs,), 
                        (ss.argus.rvs, 1,),
                        (ss.beta.rvs, 2.31, 0.63,),
                        (ss.betaprime.rvs, 5, 6,),
                        (ss.bradford.rvs, 0.3,),
                        (ss.burr.rvs, 10.5, 4.3,),
                        (ss.burr12.rvs, 10, 4,),
                        (ss.chi.rvs, 78,),
                        (ss.chi2.rvs, 55,),
                        (ss.erlang.rvs, 10,),
                        (ss.expon.rvs,),
                        (ss.exponpow.rvs, 2.7,),
                        (ss.exponweib.rvs, 2.89, 1.95,),
                        (ss.f.rvs, 29, 18,),
                        (ss.fatiguelife.rvs, 29,),
                        (ss.fisk.rvs, 3.09,),
                        (ss.foldcauchy.rvs, 4.72,),
                        (ss.foldnorm.rvs, 1.95,),
                        (ss.gamma.rvs, 1.99,),
                        (ss.gausshyper.rvs, 13.76, 3.12, 2.51, 5.18,),
                        (ss.genexpon.rvs, 9.13, 16.23, 3.28,),
                        (ss.gengamma.rvs, 4.42, 3.12,),
                        (ss.gengamma.rvs, 4.42, -3.12,),
                        (ss.genhalflogistic.rvs, 0.77,),
                        (ss.geninvgauss.rvs, 2.3, 1.5,),
                        (ss.halfgennorm.rvs, 0.67,),
                        (ss.genpareto.rvs, 0.1,),
                        (ss.gilbrat.rvs,),
                        (ss.gompertz.rvs, 0.95,),
                        (ss.halfcauchy.rvs,),
                        (ss.halflogistic.rvs,),
                        (ss.halfnorm.rvs,),
                        (ss.invgamma.rvs, 4.07,),
                        (ss.invgauss.rvs, 0.15,),
                        (ss.invweibull.rvs, 10.58,),
                        (ss.johnsonsb.rvs, 4.32, 3.18,),
                        (ss.kappa3.rvs, 1, ),
                        (ss.ksone.rvs, 1000,),
                        (ss.kstwo.rvs, 10,),
                        (ss.kstwobign.rvs,),
                        (ss.levy.rvs,),
                        (ss.loglaplace.rvs, 3.25,),
                        (ss.lognorm.rvs, 0.95,),
                        (ss.loguniform.rvs, 0.01, 1,),
                        (ss.lomax.rvs, 1.88,),
                        (ss.maxwell.rvs,),
                        (ss.mielke.rvs, 10.4, 4.6,),
                        (ss.nakagami.rvs, 4.97,),
                        (ss.ncf.rvs, 27, 27, 0.42,),
                        (ss.ncx2.rvs, 21, 1.06,),
                        (ss.pareto.rvs, 2.62,),
                        (ss.powerlaw.rvs, 1.66,),
                        (ss.powerlognorm.rvs, 2.14, 0.45,),
                        (ss.rayleigh.rvs,),
                        (ss.recipinvgauss.rvs, 0.63,),
                        (ss.reciprocal.rvs, 0.01, 1,),
                        (ss.rice.rvs, 0.77,),
                        (ss.trapz.rvs, 0.2, 0.8,),
                        (ss.triang.rvs, 0.16,),
                        (ss.truncexpon.rvs, 4.69,),
                        (ss.truncnorm.rvs, 0.1, 2,),
                        (ss.uniform.rvs,),
                        (ss.wald.rvs,),
                        (ss.weibull_min.rvs, 1.79,),
                        (ss.wrapcauchy.rvs, 0.03,)],
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

