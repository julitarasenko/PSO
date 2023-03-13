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

def pso(d, swarm_size, domain, sets, sets_j, qmc_interval, test, max_iter): # Hyper-parameters of the algorithm
    # Get all sets
    # If test the touple length and based on this the random function param are set up

    # print("d, swarm_size, domain, sets, test: ", d, swarm_size, domain, sets, test)
    # Omega
    loc, scale = 0, 1

    if len(sets[1]) == 1:
        w = sets[1][0](loc=loc, scale=scale, size=1)
    elif len(sets[1]) == 2:
        w = sets[1][0](sets[1][1], loc=loc, scale=scale, size=1)
    elif len(sets[1]) == 3:
        w = sets[1][0](sets[1][1], sets[1][2], loc=loc, scale=scale, size=1)
    elif len(sets[1]) == 4:
        w = sets[1][0](sets[1][1], sets[1][2], sets[1][3], loc=loc, scale=scale, size=1)
    else:
        w = sets[1][0](sets[1][1], sets[1][2], sets[1][3], sets[1][4], loc=loc, scale=scale, size=1)

    # Personal influence factor
    if len(sets[2]) == 1:
        phi_p = sets[2][0](loc=loc, scale=scale, size=d)
    elif len(sets[2]) == 2:
        phi_p = sets[2][0](sets[2][1], loc=loc, scale=scale, size=d)
    elif len(sets[2]) == 3:
        phi_p = sets[2][0](sets[2][1], sets[2][2], loc=loc, scale=scale, size=d)
    elif len(sets[2]) == 4:
        phi_p = sets[2][0](sets[2][1], sets[2][2], sets[2][3], loc=loc, scale=scale, size=d)
    else:
        phi_p = sets[2][0](sets[2][1], sets[2][2], sets[2][3], sets[2][4], loc=loc, scale=scale, size=d)

    # Global influence factor
    if len(sets[3]) == 1:
        phi_g = sets[3][0](loc=loc, scale=scale, size=d)
    elif len(sets[3]) == 2:
        phi_g = sets[3][0](sets[3][1], loc=loc, scale=scale, size=d)
    elif len(sets[3]) == 3:
        phi_g = sets[3][0](sets[3][1], sets[3][2], loc=loc, scale=scale, size=d)
    elif len(sets[3]) == 4:
        phi_g = sets[3][0](sets[3][1], sets[3][2], sets[3][3], loc=loc, scale=scale, size=d)
    else:
        phi_g = sets[3][0](sets[3][1], sets[3][2], sets[3][3], sets[3][4], loc=loc, scale=scale, size=d)

    # # Create particles inside the range
    if (len(sets[0]) == 1 and (sets_j < qmc_interval[0] or sets_j > qmc_interval[1])):
        X_ = sets[0][0](size=(swarm_size, d))
    else:
        X_ = sets[0][0](d=d).random(n=swarm_size)
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