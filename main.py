import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

import resultat

dim = 8 # liczba zmiennych w f. testowej
n_particle = 8 #liczbę cząstek
dims = [-100, 100] #dziedzina f. testowej
phi_p = 0.5 #skalowanie przyciągania przez najlepsze położenie lokalne cząstki
phi_g = 0.6 #skalowanie przyciągania przez cząstkę w roju najlepszą
max_iter = 500 #Ile razy wykona się przemieszczanie się roju

N = 2 #Ile razy wykona sie metodapso dla jednego rozkladu

#inicjalizacja populacja cząstek
quasirandom = qmc.Sobol(dim, scramble=True)
quasirandom = quasirandom.random(n_particle)
swarm = dims[0] + quasirandom*(dims[1] - dims[0])

resultat1 = resultat.resultat(N, dim,n_particle,swarm, 1, phi_p,phi_g,max_iter,dims)
resultat2 = resultat.resultat(N, dim,n_particle,swarm, 2, phi_p,phi_g,max_iter,dims)
resultat3 = resultat.resultat(N, dim,n_particle,swarm, 3, phi_p,phi_g,max_iter,dims)
resultat4 = resultat.resultat(N, dim,n_particle,swarm, 4, phi_p,phi_g,max_iter,dims)

plt.plot(range(max_iter), resultat1, 'k')
plt.plot(range(max_iter), resultat2, 'r')
plt.plot(range(max_iter), resultat3, 'b')
plt.plot(range(max_iter), resultat4, 'g')
plt.legend(['U[0,1]','U[-1,1]','G(0,1)','Levy'])
plt.show()