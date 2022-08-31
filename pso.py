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
        fittness <- problem to be solved
"""
import numpy as np

def pso(d, swarm_size, domain, sets, test): # Hyper-parameter of the algorithm
    # Get all sets
    # If test the touple length and based on this the random function param are set up

    # Omega
    if len(sets[1]) == 1:
        w = sets[1][0](size=1)
    else:
        w = sets[1][0](set[1][1],size=1)
    # Personal influence factor
    if len(sets[2]) == 1:
        phi_p = sets[2][0](size=d)
    else:
        phi_p = sets[2][0](sets[2][1],size=d)
    # Global influence factor
    if len(sets[3]) == 1:
        phi_g = sets[3][0](size=d)
    else:
        phi_g = sets[3][0](set[3][1], size=d)

    # print(w, phi_p, phi_g)

    # # Create particles inside the range
    if len(sets[3]) == 1:
        X_ = sets[0][0](size=(d, swarm_size))
    else:
        X_ = sets[0][0](sets[0][1], size=(d, swarm_size))
    # Transformation to a given domain
    old_min, old_max = X_.min(), X_.max()
    X = ((X_ - old_min) / (old_max - old_min)) * (domain[1] - domain[0]) + domain[0]
    # Initial velocity
    V = np.zeros((d, swarm_size)) # zero
    # V = ss.norm.rvs(d, swarm_size) * 0.1 #lub losowe z rozkładu normalnego

    # Initial bests
    pBest = X
    pBest_fit = test(X)
    gBest = pBest[:, pBest_fit.argmin()]
    gBest_fit = pBest_fit.min()

    print(gBest_fit)
    # PSO Loop