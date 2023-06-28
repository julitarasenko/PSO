import numpy as np
import math
import pandas as pd

def pso(d, swarm_size, domain, sets, sets_j, qmc_interval, test, max_iter, exp_min, iftest):
    """
    The main pso function - optimization by performing particles moving in the expected area
    :param d: problem dimensions
    :param swarm_size: number of particles
    :param domain: range of problem inputs
    :param sets: parameters for randomized tests - distributions to be tested
    :param sets_j:
    :param qmc_interval:
    :param test: test function/benchmark
    :param max_iter: maximum number of iterations
    :param exp_min: expected minimum, if it is given
    :param iftest: ????
    :return: df with results
    """
    # Create particles inside the range
    if sets_j < qmc_interval[0] or sets_j > qmc_interval[1]:
        x_ = sets[0].rvs(size=(swarm_size, d))
    else:
        x_ = sets[0](d=d).random(n=swarm_size)
    # Transformation to a given domain: using the provided X_ the range might be outside of the domain
    #
    old_min, old_max = x_.min(), x_.max()
    x = ((x_ - old_min) / (old_max - old_min)) * (domain[1] - domain[0]) + domain[0]

    # Velocity initialization
    # v = np.zeros((swarm_size, d)) # zero
    v_max = .25 * (domain[1] - domain[0])
    v = np.random.randn(swarm_size, d) * 0.1  # lub losowe z rozkładu normalnego
    # v = np.random.uniform(low=-v_max, high=v_max, size=(swarm_size,d))

    # Initial bests
    pBest = x
    pBest_fit = test(x)
    gBest = x[pBest_fit.argmin()]
    gBest_fit = pBest_fit.min()

    results = []
    # stop = 0
    # gBest_fit_last = 0

    # PSO Loop
    for j in range(max_iter):
        # v = sets[1].rvs(size=1) * v + sets[2].rvs(size=d) * (pBest - x) + sets[3].rvs(size=d) * (gBest - x)
        # v = 0.8 * v + sets[2].rvs(size=1) * (pBest - x) + sets[3].rvs(size=1) * (gBest - x)
        v = 0.8 *v + 0.1 * np.random.uniform() * (pBest - x) + 0.1 * np.random.uniform() * (gBest - x)
        x = x + v

        # Apply position bounds and count clipped values
        x_clipped = np.clip(x, domain[0], domain[1])
        x_corr = np.sum(x != x_clipped)
        x = x_clipped

        # Apply velocity bounds and count clipped values
        v_clipped = np.clip(v, -v_max, v_max)
        v_corr = np.sum(v != v_clipped)
        v = v_clipped

        # Update personal best
        fit = test(x)
        better_mask = fit < pBest_fit
        pBest[better_mask] = x[better_mask]
        pBest_fit[better_mask] = fit[better_mask]

        # Update global best
        gBest = pBest[pBest_fit.argmin()]
        gBest_fit = pBest_fit.min()

        results.append([j+1, gBest_fit, np.mean(fit), x_corr, v_corr, gBest])

        # Early stopping
        if math.isclose(gBest_fit, exp_min, abs_tol=1e-5):
            break

        # if (math.isclose(gBest_fit, gBest_fit_last, abs_tol=1e-20)):
        #     stop += 1
        # else:
        #     stop = 0
        #
        # if (set_index >= 5000 and set_index * 0.75 < stop):
        #     break

        # gBest_fit_last = gBest_fit

    pd.DataFrame(results).to_csv(f"allresultPSO-{str(test).split(' ')[1]}.csv", index=False, mode='a')

    results = np.array(results)

    return [round(results[-1][1], 5),
            round(results[:, 2].mean(), 5),
            round(results[:, 2].std(), 5),
            round(results[:, 3].mean(), 5),
            round(results[:, 4].mean(), 5),
            j + 1]
