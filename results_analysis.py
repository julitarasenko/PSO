import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from test_sets import test_sets
from distribution_sets import distribution_sets

def create_table(criteria):
    path = "./results/*.csv"
    files = glob.glob(path)
    results = []
    results_idx = 0
    for i1 in criteria['function_name']:
        for i2 in criteria['swarm_distribution']:
            for i3 in criteria['phi_p_distribution']:
                for i4 in criteria['phi_g_distribution']:
                    for i5 in criteria['experiment']:
                        criteria_arr = [i1, i2, i3, i4, i5]
                        for file in files:
                            file_name = os.path.basename(file)
                            name_without_extension = os.path.splitext(file_name)[0]
                            part_name = name_without_extension.split("_")
                            for i in range(len(criteria_arr)):
                                if criteria_arr[i] is not None:
                                    if part_name[i] == criteria_arr[i]:
                                        results.append(pd.read_csv(file))
                                        results[results_idx]['Criteria'] = part_name[i]
                                        results_idx += 1
    return pd.concat(results)

def define_func_parameters(param_ftest):
    func_all = []
    func_unimodal = []
    func_multimodal = []
    func_separable = []
    func_nonseparable = []
    for i in range(len(param_ftest['name'])):
        func_name = str(param_ftest['name'][i]).split(" ")[1]
        func_all.append(func_name)
        if param_ftest['unimodal'][i] == True:
            func_unimodal.append(func_name)
        else:
            func_multimodal.append(func_name)

        if param_ftest['separable'][i] == True:
            func_separable.append(func_name)
        else:
            func_nonseparable.append(func_name)
    return func_all, func_unimodal, func_multimodal, func_separable, func_nonseparable

def define_dist_swarm_parameters(swarm_dist):
    swarm_dist_ssqmc = [] 
    swarm_dist_ss = []
    for i in swarm_dist:
        if isinstance(i, type):
            swarm_dist_ssqmc.append(str(i).split('.')[-1][:-2])
        else: 
            swarm_dist_ss.append(i.dist.name)
    swarm_dist_all = swarm_dist_ss + swarm_dist_ssqmc
    return swarm_dist_all, swarm_dist_ssqmc, swarm_dist_ss

def define_dist_phi_parameters(generator_set):
    phi_p_dist_all = [] 
    phi_g_dist_all = []
    for phi_p in generator_set['phi_p']:
        phi_p_dist_all.append(phi_p.dist.name)
    for phi_g in generator_set['phi_g']:
        phi_g_dist_all.append(phi_g.dist.name)
    return phi_p_dist_all, phi_g_dist_all

def calculate_statistics(df):
    statistics = df.describe()
    statistics.loc['std'] = df.std()
    return statistics

def create_boxplot(df, criteria):
    title_parts = []
    for key, value in criteria.items():
        if value != [None]:
            title_parts.append(f"{key}")
    title = ', '.join(title_parts)
    for column in ['Mean_best', 'Std_best', 'Mean_f', 'Mean_acc', 'Std_scc', 'Max_acc', 'Max Iteration', 'Time']:
        df.boxplot(by = 'Criteria', column = column)  
        plt.title(column)
        plt.savefig(f'{title}_{column}.png')

def step(criteria):
    results = create_table(criteria)
    print(results)
    statistics = calculate_statistics(results)
    print(statistics)
    create_boxplot(results, criteria)


if __name__ == "__main__":
    _, param_ftest = test_sets()
    generator_set = distribution_sets()

    func_name_all, func_unimodal, func_multimodal, func_separable, func_nonseparable = define_func_parameters(param_ftest)
    swarm_dist_all, swarm_dist_ssqmc, swarm_dist_ss = define_dist_swarm_parameters(generator_set['swarm'])
    phi_p_dist_all, phi_g_dist_all = define_dist_phi_parameters(generator_set)

    criteria = {
        'function_name': None,
        'swarm_distribution': swarm_dist_all,
        'phi_p_distribution': None,
        'phi_g_distribution': None,
        'experiment': None,
            }

    for key, value in criteria.items():
        if value is None:
            criteria[key] = [None]

    step(criteria)


