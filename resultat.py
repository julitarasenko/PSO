import numpy as np

import analiz
import metodapso

def resultat(N, dim,n_particle,swarm, metoda, phi_p,phi_g,max_iter,dims):
    resultat = np.zeros(max_iter)
    for i in range(N):
        swarm1, results = metodapso.metodapso(dim,n_particle,swarm, metoda, phi_p,phi_g,max_iter,dims)
        if i==0:
            g_best = results[max_iter-1]
            results = results.transpose()
            resultat = results
        else:
            g_best = np.vstack((g_best, results[max_iter-1]))
            results = results.transpose()
            resultat = np.vstack((resultat, results))
    resultat = sum(resultat)/len(g_best)
    srednie, sigma, lepszy, gorszy = analiz.analiz(g_best)
    print(metoda)
    print(srednie)
    print(sigma)
    print(lepszy)
    print(gorszy)
    return resultat