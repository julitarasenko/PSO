import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from test_sets import test_sets
from distribution_sets import distribution_sets

def create_table(cr):
    path = "./results/*.csv"
    files = glob.glob(path)
    results = []
    results_idx = 0
    criteria = cr.copy()
    for key, value in criteria.items():
        if not isinstance(value, list):
            criteria[key] = [value]
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
                            part_name = weibull_check(part_name)
                            criteria_name = create_criteria_name(criteria_arr)
                            if all(part_name[i] == criteria_arr[i] for i in range(len(criteria_arr)) 
                                if criteria_arr[i] is not None):
                                results.append(pd.read_csv(file))
                                results[results_idx]['Distribution'] = criteria_name
                                results_idx += 1
    return pd.concat(results)

def create_criteria_name(criteria_arr):
    for i in range(len(criteria_arr)):
        if criteria_arr[i] is not None:
            criteria_name = criteria_arr[i]
    return criteria_name

def weibull_check(part_name):
    if part_name[2] == 'weibull' or part_name[4] == 'weibull':
        part_name_weibull = [part_name[0], part_name[1], 'weibull_min', 'weibull_min', 'total']
        if part_name[2] != 'weibull':
            part_name_weibull[2] = part_name[2]
        if part_name[4] != 'weibull':
            part_name_weibull[3] = part_name[4]
        part_name = part_name_weibull
    return part_name

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
    phi_dist_all = [] 
    for phi in generator_set['phi_p']:
        phi_dist_all.append(phi.dist.name)
    return phi_dist_all

def calculate_statistics(df):
    column = ['Mean_best', 'Mean_acc', 'Max Iteration']
    for i in column:
        statistics = df.groupby('Distribution')[i].describe()
        if i != 'Mean_acc':
            statistics['Criteria'] = statistics['mean'].rank(ascending=False, pct=True)
        else:
            statistics['Criteria'] = statistics['mean'].rank(ascending=True, pct=True)
        statistics.sort_values(by='Criteria', ascending=False, inplace=True)
        # if i == 'Mean_best':
        #     Mean_best = statistics
        # if i == 'Mean_acc':
        #     Mean_acc = statistics
        # if i == 'Max Iteration':
        #     Max_iter = statistics
        pd.set_option('display.max_rows', None)
        print(statistics)
        print()
    
    # top = Max_iter + Mean_best + Mean_acc
    # top.sort_values(by='Criteria', ascending=False, inplace=True)
    # pd.set_option('display.max_rows', None)
    # print(top)


def create_boxplot(df, criteria):
    func = criteria['function_name']

    for key, value in criteria.items():
        if value is not None:
            criteria_name = key

    criteria_list = np.array(df['Distribution'].unique())
    index_list = []
    for i in range(1, len(criteria_list) + 1):
        index_list.append(i)
    column = ['Mean_best', 'Mean_acc', 'Max Iteration']   
    column_name = ['mB', 'mACC', 'mFE'] 

    for i in range(len(column)): 
        if criteria['phi_p_distribution'] is not None or criteria['phi_g_distribution'] is not None:
            ax = df.boxplot(by = 'Distribution', column = column[i], fontsize = 14, figsize = (21, 3))
        else:
            ax = df.boxplot(by = 'Distribution', column = column[i], fontsize = 14, figsize = (3, 3))
        ax.set_ylabel(column_name[i], fontsize = 18)
        ax.set_xlabel('Distribution', fontsize = 18)
        plt.xticks(index_list, index_list)
        plt.suptitle('')
        plt.title(f'{func}', fontsize = 18)
        plt.savefig(f'{func}_{criteria_name}_{column_name[i]}.png', dpi=300, bbox_inches = "tight")


def step(criteria):
    results = create_table(criteria)
    results['Max Iteration'] *= 20
    print(results)
    calculate_statistics(results)
    create_boxplot(results, criteria)


if __name__ == "__main__":
    _, param_ftest = test_sets()
    generator_set = distribution_sets()

    func_name_all, func_unimodal, func_multimodal, func_separable, func_nonseparable = define_func_parameters(param_ftest)
    swarm_dist_all, swarm_dist_ssqmc, swarm_dist_ss = define_dist_swarm_parameters(generator_set['swarm'])
    phi_dist_all = define_dist_phi_parameters(generator_set)

    criteria = {
        'function_name': 'spheref',
        'swarm_distribution': swarm_dist_all,
        'phi_p_distribution': None,
        'phi_g_distribution': None,
        'experiment': None,
            }

    step(criteria)


