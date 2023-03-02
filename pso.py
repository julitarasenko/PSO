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

def pso(d, swarm_size, domain, w, phi_p, phi_g, X_, test, max_iter): # Hyper-parameters of the algorithm
    
    # Transformation to a given domain
    old_min, old_max = X_.min(), X_.max()
    X = ((X_ - old_min) / (old_max - old_min)) * (domain[1] - domain[0]) + domain[0]
    corr=0
    for i in range(d):
        for j in range(swarm_size):
            if (X[j, i] > domain[1]  or X[j, i] < domain[0]):
                print("swarm, min, max: ", X[j, i], domain[0], domain[1] )
                corr+=1
    if (corr==0):
        print("Swarm jest w prawidłowych przedziałach")
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
    return [results[-1][1], np.array(results).mean(axis=0)[2],
            np.array(results).std(axis=0)[2], np.array(results).mean(axis=0)[3],
            np.array(results).mean(axis=0)[4] ] # X, results