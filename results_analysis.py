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

    _, param_ftest = test_sets()
    generator_set = distribution_sets()
    swarm_dist = generator_set['swarm']
    swarm_dist_ssqmc_name = []
    for i in swarm_dist:
        if isinstance(i, type):
            swarm_dist_ssqmc_name.append(str(i).split('.')[-1][:-2])

    for file in files:
        file_name = os.path.basename(file)
        name_without_extension = os.path.splitext(file_name)[0]
        part_name = name_without_extension.split("_")
        dict_file_name = {
            'function_name': part_name[0],
            'swarm_distribution': part_name[1],
            'phi_p_distribution': part_name[2],
            'phi_g_distribution': part_name[3],
            'experiment': part_name[4]
            }
        add_function_parameters(dict_file_name, param_ftest)
        add_distribution_parameters(dict_file_name, swarm_dist_ssqmc_name)

        if all(dict_file_name[criteria_name] == criteria_value for criteria_name, criteria_value in criteria.items() 
               if criteria_value is not None):
            results.append(pd.read_csv(file))
    return pd.concat(results)

def add_function_parameters(dict, param_ftest):
    for i in range(len(param_ftest['name'])):
        if dict['function_name'] == str(param_ftest['name'][i]).split(" ")[1]:
            dict['function_unimodal'] = param_ftest['unimodal'][i]
            dict['function_separable'] = param_ftest['separable'][i]
            break

def add_distribution_parameters(dict, swarm_dist_ssqmc_name):
    dict['swarm_distribution_qmc'] = False
    for i in swarm_dist_ssqmc_name:
        if dict['swarm_distribution'] == i:
            dict['swarm_distribution_qmc'] = True
            break

def calculate_statistics(df):
    statistics = df.describe()
    statistics.loc['std'] = df.std()
    return statistics

def create_boxplot(df, criteria):
    title_parts = []
    for key, value in criteria.items():
        if value is not None:
            title_parts.append(f"{key}={value}")

    title = ', '.join(title_parts)

    df.boxplot(column=['Mean_best', 'Std_best', 'Mean_f', 'Mean_acc', 'Std_scc', 'Max_acc', 'Time'])  
    plt.title(title)
    plt.savefig(f'{title}.png')

def step(criteria):
    results = create_table(criteria)
    print(results)
    statistics = calculate_statistics(results)
    print(statistics)
    create_boxplot(results, criteria)


if __name__ == "__main__":
    criteria = {
        'function_name': None,
        'function_unimodal': None,
        'function_separable': None,
        'swarm_distribution': None,
        'swarm_distribution_qmc': True,
        'phi_p_distribution': None,
        'phi_g_distribution': None,
        'experiment': None,
            }

    step(criteria)

