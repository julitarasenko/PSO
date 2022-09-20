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

def pso(d, swarm_size, domain, sets, test): # Hyper-parameter of the algorithm
    # Get all sets
    # If test the touple length and based on this the random function param are set up

    # print("d, swarm_size, domain, sets, test: ", d, swarm_size, domain, sets, test)
    # Omega
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
    if len(sets[0]) == 1:
        X_ = sets[0][0](size=(swarm_size, d))
    else:
        X_ = sets[0][0](sets[0][1], size=(swarm_size, d))
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

    max_iter = 5  # !!!!Set this as the parameter in halving!!
    results = []

    # PSO Loop
    for j in range(max_iter):
        x_corr = 0
        v_corr = 0
        V = w * V + phi_p * (pBest - X) + phi_g * (gBest - X)
        X = X + V

        # position outside the domain - correct moving to the domain edge
        if ( any(X[(X < domain[0]) | (X > domain[1])])):
            X[X < domain[0]] = domain[0]
            X[X > domain[1]] = domain[1]
            x_corr += len(X[(X < domain[0]) | (X > domain[1])])
        # max_vel control
        if( any(V[(V < -V_max) | (V > V_max)])):
            V[V < -V_max] = -V_max
            V[V > V_max] = V_max
            v_corr += len(V[(V < -V_max) | (V > V_max)])

        # personal best setup
        fit = test(X)
        pBest[(pBest_fit >= fit), :] = X[(pBest_fit >= fit), :]
        pBest_fit = np.array([pBest_fit, fit]).min(axis=0)

        # global best
        gBest = pBest[pBest_fit.argmin()]
        gBest_fit = pBest_fit.min()

        results.append([gBest, gBest_fit, np.average(fit), x_corr, v_corr])
    return X, results