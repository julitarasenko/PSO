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

def pso_domain(d, swarm_size, domain, w, phi_p, phi_g, X_, test, max_iter): # Hyper-parameter of the algorithm
    
    # Transformation to a given domain
    old_min, old_max = X_.min(), X_.max()
    X = np.zeros((swarm_size, d))
    V_max = 1/3 * abs(domain[0][1] - domain[0][0])
    for i in range(d):
        X[:, i] = ((X_[:, i] - old_min) / (old_max - old_min)) * (domain[i][1] - domain[i][0]) + domain[i][0]
        if (1/3 * abs(domain[i][1] - domain[i][0]) > V_max): 
            V_max = 1/3 * abs(domain[i][1] - domain[i][0])
       
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
        for i in range(d):
            X_ = X[:, i]
            if ( any(X_[(X_ < domain[i][0]) | (X_ > domain[i][1])])):
                x_corr += len(X_[(X_ < domain[i][0]) | (X_ > domain[i][1])])
                X_[X_ < domain[i][0]] = domain[i][0]
                X_[X_ > domain[i][1]] = domain[i][1]
            X[:, i] = X_

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
            np.array(results).mean(axis=0)[4]]  # X, results