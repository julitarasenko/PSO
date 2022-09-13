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
from scipy.stats import qmc

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
    V = np.zeros((swarm_size, d)) # zero
    # V = ss.norm.rvs(d, swarm_size) * 0.1 #lub losowe z rozkładu normalnego

    # Initial bests
    pBest = np.zeros(d)
    pBest_fit = np.ones(swarm_size) * np.inf
    gBest_fit = np.inf
    gBest = np.zeros(d)

    max_iter = 5 #Ile razy wykona się przemieszczanie się roju
    results = np.zeros((max_iter,3))
    sumcorrection = 0

    # PSO Loop
    for j in range(max_iter):
        gBest_fit = pBest_fit.min()
        gBest = X[pBest_fit.argmin()]
        
        #obliczanie wartości cząstki wg zadanej funkcji
        pBest_fit = test(X)
        pBest = X[pBest_fit.argmin()]

        for i in range(swarm_size):
            V[i] = w * V[i] + phi_p * np.random.rand(1,d).dot(pBest - X[i]) + phi_g * np.random.rand(1,d).dot(gBest - X[i])
            for n in range(d):
                if V[i][n] > 1/3 * abs(domain[1] - domain[0]):
                    V[i][n] = 1/3 * abs(domain[1] - domain[0])
            # sprawdzenie rozpędzającej się cząstki i ustawić na wartość
            # max_vel
            X[i] = X[i] + V[i] 

            for n in range(d):
                if X[i][n] < domain[0]:
                    X[i][n] = domain[0]
                    sumcorrection += 1
                if X[i][n] > domain[1]:
                    X[i][n] = domain[1]
                    sumcorrection += 1
            # kontrola położenia cząstki w założonej dziedzinie, jeżeli poza,
            # to ustawić na wartość odpowiednio min lub max

        results[j] = [gBest_fit, max_iter, sumcorrection]
    return X, results