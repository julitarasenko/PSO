import warnings
from Run_optimization import run_optimization
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from test_sets import test_sets
import numpy as np
import csv
from distribution_sets import distribution_sets
from itertools import product

def execute_pso(func, bound, dim, min_value, x_opt, iterations, swarm_dist, dist1, dist2, d_type):
    func_best_scores = []
    func_average_swarm = []
    func_average_best = []
    func_std_dev_best = []
    func_max_iters = []
    func_average_acc = []
    func_std_acc = []
    func_max_acc = []
    func_time = []

    # Prepare parameters
    params = [(dim, bound, func, min_value, iterations, x_opt, swarm_dist, dist1, dist2, d_type)] * n_runs

    # Create a ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=cores) as executor:
        results = executor.map(run_optimization, params)

    # print(results)
    (func_best_scores, func_average_best, func_std_dev_best, func_average_swarm,
     func_average_acc, func_std_acc, func_max_acc, func_max_iters, func_time) = zip(*results)

    # print(results)
    # print(f'Function: {func.__name__}\n' ,
    #   f' Best position: {best_position}\n' ,
    #   f' Itaration with the solution: {max_iter}\n',
    #   f' Best score: {best_score},\n',
    #   f' Distance {best_score - min_value}')

    # Compute and store averages
    avg_best_score = np.mean(func_best_scores)
    avg_swarm = np.mean(func_average_swarm)
    avg_max_iter = np.mean(func_max_iters)

    # print(f'Function: {func.__name__}\n',
    #       f' Best score: {avg_best_score},\n',
    #       f' Average swarm: {avg_swarm},\n',
    #       f' Average max iteration: {avg_max_iter}')

    # Write results to a CSV file
    labels = ['Best Score', 'Mean_best', 'Std_best', 'Mean_f', 'Mean_acc', 'Std_scc', 'Max_acc', 'Max Iteration',
              'Time']
    if isinstance(swarm_dist, type):
        swarm_dist_name = str(swarm_dist).split('.')[-1][:-2]
    else:
        swarm_dist_name = swarm_dist.dist.name

    with open(f'results/{func.__name__}_{swarm_dist_name}_{dist1.dist.name}_{dist2.dist.name}_total.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(labels)
        for best_score, average_best, std_dev, average_func, avg_acc, std_acc, max_acc, max_iter, time in \
                zip(func_best_scores, func_average_best, func_std_dev_best, func_average_swarm,
                    func_average_acc, func_std_acc, func_max_acc, func_max_iters, func_time):
            writer.writerow(
                [best_score, average_best, std_dev, average_func, avg_acc, std_acc, max_acc, max_iter, time])

    # # Plotting the scores
    # plt.figure(figsize=(12, 6))
    # plt.plot(pso.best_scores_history, label='Best score')
    # plt.plot(pso.average_scores_history, label='Average score')
    # plt.legend()
    # plt.xlabel('Iteration')
    # plt.ylabel('Score')
    # plt.title(f'Best and Average Score of the Swarm Over Iterations for {func.__name__} Function')
    # plt.show()


# Main code execution
if __name__ == "__main__":
    # Settings the warnings to be ignored
    warnings.filterwarnings('ignore')

    # PSO parameters
    _, test_functions = test_sets()
    iterations = 100  # Number of iterations
    n_runs = 10  # Number of times to run the optimization for each function
    cores = 4  # Number of cores to use
    # Define the desired dtype
    d_type = np.float128

    generator_set = distribution_sets()

    # Generate all pairs of distributions
    combinations = product(generator_set['swarm'], generator_set['phi_p'], generator_set['phi_g'])

    # Optimization process for each function
    for func, bound, dim, min_value, x_opt in \
            zip(test_functions['name'], test_functions['domain'], test_functions['dim'],
                test_functions['min'], test_functions['x_best']):
        for swarm_dist, dist1, dist2 in combinations:
            # if isinstance(swarm_dist, type):
            #     print(str(swarm_dist).split('.')[-1][:-2], dist1.dist.name, dist2.dist.name)
            # else:
            #     print(swarm_dist.dist.name, dist1.dist.name, dist2.dist.name)
            execute_pso(func, bound, dim, min_value, x_opt, iterations, swarm_dist, dist1, dist2, d_type)


