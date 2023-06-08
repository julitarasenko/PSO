import warnings
from Run_optimization import run_optimization
from concurrent.futures import ProcessPoolExecutor
from test_sets import test_sets
import numpy as np
from distribution_sets import distribution_sets
import pandas as pd
import time

generator_set = distribution_sets()
_, param_ftest = test_sets()
cores = 15
d_type = np.float128

def execute_algorithm(func, dim, bound, min_value, x_opt):
    warnings.filterwarnings('ignore')
    max_iter = 10000  # Number of iterations

    setup = [] # store all combinations of 4 params combinations
    for i1 in generator_set['swarm']:
        for i2 in generator_set['phi_p']:
            for i3 in generator_set['phi_g']:
                    setup.append([i1, i2, i3])

    df_result = pd.DataFrame(columns=["setIdx", "Swarm", "phiP", "phiG",
                                      "MaxIter", "BestScore", "Mean_best",
                                      "Std_best", "Mean_f", "Mean_acc", "Std_acc",
                                      "Max_acc", "Iteration", "Time",])
    df_result.to_csv(f"result-{str(func).split(' ')[1]}.csv", index=False)
    
    for j in range(len(setup)):
        swarm_dist = setup[j][0]
        dist1 = setup[j][1]
        dist2 = setup[j][2]

        params = [(dim, bound, func, min_value, max_iter, x_opt, swarm_dist, dist1, dist2, d_type)]
        results = run_optimization(params)
            
        if isinstance(swarm_dist, type):
            swarm_dist_name = str(swarm_dist).split('.')[-1][:-2]
        else:
            swarm_dist_name = swarm_dist.dist.name
        
        phi_p_dist_name = dist1.dist.name
        phi_g_dist_name = dist2.dist.name

        df_result.loc[len(df_result)] = [j, swarm_dist_name, phi_p_dist_name, 
                                            phi_g_dist_name, max_iter] + results
        df_new_row = pd.DataFrame([[j, swarm_dist_name, phi_p_dist_name, 
                                    phi_g_dist_name, max_iter] + results], 
                                    columns=df_result.columns)
        df_new_row.to_csv(f"result-{str(func).split(' ')[1]}.csv", mode='a', header=False, index=False) 

def test(i):
    func = param_ftest['name'][i]
    dim = param_ftest['dim'][i]
    bound = param_ftest['domain'][i]
    min_value = param_ftest['min'][i]
    x_opt = param_ftest['x_best'][i]
    execute_algorithm(func, dim, bound, min_value, x_opt)


def main():
    start = time.time()
    with ProcessPoolExecutor(max_workers=cores) as executor:
        executor.map(test, range(len(param_ftest['name'])))

    end = time.time()
    print('{:.4f} s'.format(end-start))

if __name__ == "__main__":
    main()
