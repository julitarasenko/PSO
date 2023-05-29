from scipy import stats as ss

loc, scale = 0, 1

def distribution_sets():
    generator_set = {
            'swarm': [  
                        ss.norm(loc=loc, scale=scale),
                        ss.uniform(loc=loc, scale=scale),
                        ss.levy(loc=loc, scale=scale),
                        ss.qmc.Sobol, #start library scipy.stats.qmc in index 3
                        ss.qmc.Halton, 
                        ss.qmc.LatinHypercube, #end ibrary scipy.stats.qmc in index 5
                        # # (ss.qmc.PoissonDisk,)
                    ],

            'omega': [  
                        ss.alpha(3.57, loc=loc, scale=scale),
                        ss.arcsine(loc=loc, scale=scale),
                        ss.argus(1, loc=loc, scale=scale),
                        ss.beta(2.31, 0.63, loc=loc, scale=scale),
                        ss.betaprime(5, 6, loc=loc, scale=scale),
                        ss.bradford(0.3, loc=loc, scale=scale),
                        ss.burr(10.5, 4.3, loc=loc, scale=scale),
                        ss.burr12(10, 4, loc=loc, scale=scale),
                        ss.chi(78, loc=loc, scale=scale),
                        ss.chi2(55, loc=loc, scale=scale),
                        ss.erlang(10, loc=loc, scale=scale),
                        ss.expon(loc=loc, scale=scale),
                        ss.exponpow(2.7, loc=loc, scale=scale),
                        ss.exponweib(2.89, 1.95, loc=loc, scale=scale),
                        ss.f(29, 18, loc=loc, scale=scale),
                        ss.fatiguelife(29, loc=loc, scale=scale),
                        ss.fisk(3.09, loc=loc, scale=scale),
                        ss.foldcauchy(4.72, loc=loc, scale=scale),
                        ss.foldnorm(1.95, loc=loc, scale=scale),
                        ss.gamma(1.99, loc=loc, scale=scale),
                        # ss.gausshyper(13.76, 3.12, 2.51, 5.18, loc=loc, scale=scale),
                        # ss.genexpon(9.13, 16.23, 3.28, loc=loc, scale=scale),
                        ss.gengamma(4.42, 3.12, loc=loc, scale=scale),
                        ss.gengamma(4.42, -3.12, loc=loc, scale=scale),
                        ss.genhalflogistic(0.77, loc=loc, scale=scale),
                        ss.geninvgauss(2.3, 1.5, loc=loc, scale=scale),
                        ss.halfgennorm(0.67, loc=loc, scale=scale),
                        ss.genpareto(0.1, loc=loc, scale=scale),
                        ss.gilbrat(loc=loc, scale=scale),
                        ss.gompertz(0.95, loc=loc, scale=scale),
                        ss.halfcauchy(loc=loc, scale=scale),
                        ss.halflogistic(loc=loc, scale=scale),
                        ss.halfnorm(loc=loc, scale=scale),
                        ss.invgamma(4.07, loc=loc, scale=scale),
                        ss.invgauss(0.15, loc=loc, scale=scale),
                        ss.invweibull(10.58, loc=loc, scale=scale),
                        ss.johnsonsb(4.32, 3.18, loc=loc, scale=scale),
                        ss.kappa3(1, loc=loc, scale=scale),
                        # ss.ksone(1000, loc=loc, scale=scale),
                        # ss.kstwo(10, loc=loc, scale=scale),
                        ss.kstwobign(loc=loc, scale=scale),
                        ss.levy(loc=loc, scale=scale),
                        ss.loglaplace(3.25, loc=loc, scale=scale),
                        ss.lognorm(0.95, loc=loc, scale=scale),
                        ss.loguniform(0.01, 1, loc=loc, scale=scale),
                        ss.lomax(1.88, loc=loc, scale=scale),
                        ss.maxwell(loc=loc, scale=scale),
                        ss.mielke(10.4, 4.6, loc=loc, scale=scale),
                        ss.nakagami(4.97, loc=loc, scale=scale),
                        ss.ncf(27, 27, 0.42, loc=loc, scale=scale),
                        ss.ncx2(21, 1.06, loc=loc, scale=scale),
                        ss.pareto(2.62, loc=loc, scale=scale),
                        ss.powerlaw(1.66, loc=loc, scale=scale),
                        ss.powerlognorm(2.14, 0.45, loc=loc, scale=scale),
                        ss.rayleigh(loc=loc, scale=scale),
                        ss.recipinvgauss(0.63, loc=loc, scale=scale),
                        ss.reciprocal(0.01, 1, loc=loc, scale=scale),
                        ss.rice(0.77, loc=loc, scale=scale),
                        ss.trapz(0.2, 0.8, loc=loc, scale=scale),
                        ss.triang(0.16, loc=loc, scale=scale),
                        ss.truncexpon(4.69, loc=loc, scale=scale),
                        ss.truncnorm(0.1, 2, loc=loc, scale=scale),
                        ss.uniform(loc=loc, scale=scale),
                        ss.wald(loc=loc, scale=scale),
                        ss.weibull_min(1.79, loc=loc, scale=scale),
                        ss.wrapcauchy(0.03, loc=loc, scale=scale)
                        ],
        }
    
    generator_set.update({'phi_p': generator_set['omega']})
    generator_set.update({'phi_g': generator_set['omega']})

    qmc_start_index = 3
    qmc_end_index = 5

    qmc_interval = [len(generator_set['omega']) * len(generator_set['phi_p']) * len(generator_set['phi_g'])*qmc_start_index,
                    len(generator_set['omega']) * len(generator_set['phi_p']) * len(generator_set['phi_g'])*(qmc_end_index+1)-1]
    
    return generator_set, qmc_interval