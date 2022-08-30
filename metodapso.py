import numpy as np
import math
from scipy.stats import qmc, levy

import plots
import problem1

def metodapso(dim,n_particle,swarm,om, phi_p,phi_g,max_iter,dims):
    omega = 0.3
    velocity = qmc.Sobol(dim, scramble=True)
    velocity = velocity.random(n_particle)
    fitness = np.zeros((n_particle,1))
    results = np.zeros((max_iter,1))

    g_best = math.inf
    g_best_particle = np.zeros(dim)
    p_best_particle = np.zeros(dim)
    p_best = np.ones(n_particle) * math.inf
    sum=0
    # główna pętla
    for j in range(max_iter):
        for i in range(n_particle):
            # znajdowanie najlepszej cząstki w roju
            if p_best[i] < g_best:
                g_best = p_best[i]
                g_best_particle = swarm[i]
        #obliczanie wartości cząstki wg zadanej funkcji
        for i in range(n_particle):
            fitness[i] = problem1.problem1(swarm[i,:])
            if fitness[i] < p_best[i]:
               p_best[i] = fitness[i]
               p_best_particle = swarm[i]
            if om==1:
                omega = np.random.uniform(0,1) 
            if om==2:
                omega = np.random.uniform(-1,1)
            if om==3:
                omega = np.random.normal(0,1)
            if om==4:
                omega = levy.pdf(1, 1, 1.4)
            quasirandom = qmc.Sobol(dim, scramble=True)
            quasirandom = quasirandom.random(1)
            velocity[i] = omega * velocity[i] + phi_p * quasirandom.dot(p_best_particle - swarm[i]) + phi_g * quasirandom.dot(g_best_particle - swarm[i])
            if velocity[i][0] > 1/3 * abs(dims[1] - dims[0]):
                velocity[i][0] = 1/3 * abs(dims[1] - dims[0])
            if velocity[i][1] > 1/3 * abs(dims[1] - dims[0]):
                velocity[i][1] = 1/3 * abs(dims[1] - dims[0])
            # sprawdzenie rozpędzającej się cząstki i ustawić na wartość
            # max_vel
            swarm[i] = swarm[i] + velocity[i]  
            if swarm[i][0] < dims[0]:
                swarm[i] = dims[0]
            if swarm[i][0] > dims[1]:
                swarm[i] = dims[1]
            if swarm[i][1] < dims[0]:
                swarm[i] = dims[0]
            if swarm[i][1] > dims[1]:
                swarm[i] = dims[1]
            # kontrola położenia cząstki w założonej dziedzinie, jeżeli poza,
            # to ustawić na wartość odpowiednio min lub max
            sum = sum + g_best
        results[j] = g_best
    return swarm, results