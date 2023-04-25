"""PSO function
    :parameter
        d <- dimentions
        swarm_size <- number of particles
        phi_p <- personal weight
        phi_g <- global weight
        w <- omega
        sets <- dictionary with keys:
            swarm:  list of tuples with random swarm initialization
            phi: list of tuples with random phi_p and phi_g initialization
            v: velocity initialization
        fitness <- problem to be solved
"""
import numpy as np
import math

def pso(d, swarm_size, domain, sets, sets_j, qmc_interval, test, max_iter, exp_min, iftest): # Hyper-parameters of the algorithm
    # Get all sets
    # If test the touple length and based on this the random function param are set up

    # print("d, swarm_size, domain, sets, test: ", d, swarm_size, domain, sets, test)
    
    w = sets[1].rvs(size=1)
    phi_p = sets[2].rvs(size=d)
    phi_g = sets[3].rvs(size=d)

    # # Create particles inside the range
    if (sets_j < qmc_interval[0] or sets_j > qmc_interval[1]):
        X_ = sets[0].rvs(size=(swarm_size, d))
    else:
        X_ = sets[0][0](d=d).random(n=swarm_size)
    # Transformation to a given domain
    old_min, old_max = X_.min(), X_.max()
    X = ((X_ - old_min) / (old_max - old_min)) * (domain[1] - domain[0]) + domain[0]
    
    # Initial velocity
    # V = np.zeros((swarm_size, d)) # zero
    V_max = 1/3 * abs(domain[1] - domain[0])
    V = np.random.randn(swarm_size, d) * V_max #lub losowe z rozkładu normalnego

    # Initial bests
    pBest = X
    pBest_fit = test(X)
    gBest = X[pBest_fit.argmin()]
    gBest_fit = pBest_fit.min()

    results = []
    stop = 0
    gBest_fit_last = 0
    # PSO Loop
    for j in range(max_iter):
        x_corr = 0
        v_corr = 0
        V = w * V + phi_p * (pBest - X) + phi_g * (gBest - X)
        X = X + V

        # position outside the domain - correct moving to the domain edge
        if ( any(X[(X < domain[0]) | (X > domain[1])])):
            x_corr += len(X[(X < domain[0]) | (X > domain[1])])
            X[X < domain[0]] = domain[0]
            X[X > domain[1]] = domain[1]

        # max_vel control
        if( any(V[(V < -V_max) | (V > V_max)])):
            v_corr += len(V[(V < -V_max) | (V > V_max)])
            V[V < -V_max] = -V_max
            V[V > V_max] = V_max

        # personal best setup
        fit = test(X)
        pBest[(pBest_fit >= fit), :] = X[(pBest_fit >= fit), :]
        pBest_fit = np.array([pBest_fit, fit]).min(axis=0)

        # global best
        gBest = pBest[pBest_fit.argmin()]
        gBest_fit = pBest_fit.min()

        results.append([j, gBest_fit, np.average(fit), x_corr, v_corr])

        if (iftest and math.isclose(gBest_fit, exp_min, abs_tol=1e-5)):
            break

        if (not(iftest) and math.isclose(gBest_fit, gBest_fit_last, abs_tol=1e-20)):
            stop += 1
        else:
            stop = 0

        if (not(iftest) and j >= 100 and j * 0.75 < stop):
            break

        gBest_fit_last = gBest_fit

    return [round(results[-1][1], 5), round(np.array(results).mean(axis=0)[2], 5),
            round(np.array(results).std(axis=0)[2], 5), round(np.array(results).mean(axis=0)[3], 5),
            round(np.array(results).mean(axis=0)[4],5), j ] # X, results