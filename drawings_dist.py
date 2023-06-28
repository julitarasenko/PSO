import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import stats as ss

def draw_figure(t, X, dim, filename):
    n1, ctr1 = np.histogram(X[:, 0], bins=20)
    n2, ctr2 = np.histogram(X[:, 1], bins=20)

    fig, ax = plt.subplots(2, 2)

    # Subplot (2, 2)
    ax[1, 1].scatter(X[:, 0], X[:, 1])
    ax[1, 1].axis([dim[0], dim[1], dim[0], dim[1]])
    ax[1, 1].set_title(t, fontsize=12, fontweight='normal')
    ax[1, 1].set_xlabel('X1')
    ax[1, 1].set_ylabel('X2')

    # Subplot (2, 4)
    ax[1, 0].bar(ctr1[:-1], -n1, width=np.diff(ctr1), align='edge')
    ax[1, 0].axis([dim[0], dim[1], -max(n1)*1.1, 0])
    ax[1, 0].axis('off')

    # Subplot (2, 1)
    ax[0, 0].barh(ctr2[:-1], -n2, height=np.diff(ctr2), align='edge')
    ax[0, 0].axis([-max(n2)*1.1, 0, dim[0], dim[1]])
    ax[0, 0].axis('off')

    # Adjust the positions of the plots to mimic the MATLAB version
    ax[1, 1].set_position([0.35, 0.35, 0.55, 0.55])
    ax[1, 0].set_position([0.35, 0.1, 0.55, 0.15])
    ax[0, 0].set_position([0.1, 0.35, 0.15, 0.55])

    # Set colormap similar to MATLAB's
    plt.set_cmap('Blues')

    plt.savefig(filename)
    plt.show()
    plt.close(fig)


def scale_val(x_, bounds):
    old_min, old_max = x_.min(), x_.max()
    return ((x_ - old_min) / (old_max - old_min)) * (bounds[1] - bounds[0]) + bounds[0]

n = 20  # number of elements
d = 2  # dimension
i = 0  # starting index in python
dim = [-5, 10]  # dimensions
loc = 0.0
scale = 1.0


# Uniform distribution
X = ss.uniform(loc=loc, scale=scale).rvs(size=(n, d))
X_s = scale_val(X, dim)


draw_figure('Uniform distribution loc=0, scale=1', X_s, dim, 'Uniform.png')

# Normal distribution
X = ss.norm(loc=loc, scale=scale).rvs(size=(n, d))
X_s = scale_val(X, dim)


draw_figure('Normal distribution loc=0, scale=1', X_s, dim, 'Normal.png')


# levy distribution
X = ss.levy(loc=loc, scale=scale).rvs(size=(n, d))
X_s = scale_val(X, dim)


draw_figure('Levy distribution loc=0, scale=1', X_s, dim, 'Levy.png')


# sobol distribution
X = ss.qmc.Sobol(d=d).random(n=n)
X_s = ss.qmc.scale(X, l_bounds=[dim[0]]*d, u_bounds=[dim[1]]*2)


draw_figure('Sobol quasi random distribution', X_s, dim, 'Sobol.png')

# Halton distribution

X = ss.qmc.Halton(d=d).random(n=n)
X_s = ss.qmc.scale(X, l_bounds=[dim[0]]*d, u_bounds=[dim[1]]*2)


draw_figure('Halton quasi random distribution', X_s, dim, 'Halton.png')

# Latin distribution

X = ss.qmc.LatinHypercube(d=d).random(n=n)
X_s = ss.qmc.scale(X, l_bounds=[dim[0]]*d, u_bounds=[dim[1]]*2)


draw_figure('LatinHypercube quasi random distribution', X_s, dim, 'LatinHypercube.png')