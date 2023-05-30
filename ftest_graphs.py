import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from test_sets import test_sets

param_problem, param_ftest = test_sets()

for i in range(np.size(param_ftest['name'])):
    # Pobranie parametrów dla i-tej funkcji
    name = param_ftest['name'][i]
    domain = param_ftest['domain'][i]
    if (name == param_ftest['name'][8]):
        domain = [-2, 2]

    # Generowanie danych
    x = np.linspace(domain[0], domain[1], 10000)
    y = np.linspace(domain[0], domain[1], 10000)
    X, Y = np.meshgrid(x, y)
    Z = name(np.column_stack((X.flatten(), Y.flatten()))).reshape(X.shape)

    # Znalezienie minimum
    min_index = np.unravel_index(np.argmin(Z), Z.shape)
    x_min = round(X[min_index], 3)
    y_min = round(Y[min_index], 3)
    min_value = round(Z[min_index], 3)

    # Tworzenie wykresu 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)

    # Konfiguracja osi
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')

    # Dodawanie etykiet
    ax.text2D(0.05, 1, name.__name__, transform=ax.transAxes, fontsize=10)
    ax.text2D(0.05, 0.95, f'Minimum: {min_value}', transform=ax.transAxes, fontsize=10)
    ax.text2D(0.05, 0.90, f'Punkt minimum: ({x_min:.2f}, {y_min:.2f})', transform=ax.transAxes, fontsize=10)

    # Zapisanie wykresu do pliku
    plt.savefig(f'{name.__name__}.png')
    
