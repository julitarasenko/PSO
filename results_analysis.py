import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_table(criteria):
    path = "./results/*.csv"
    files = glob.glob(path)
    results = []

    for file in files:
        file_name = os.path.basename(file)
        name_without_extension = os.path.splitext(file_name)[0]
        part_name = name_without_extension.split("_")
        if all(part_name[i] == criteria[key] for i, key in enumerate(criteria.keys()) if criteria[key] is not None):
            results.append(pd.read_csv(file))
    return pd.concat(results)

def calculate_statistics(df):
    statistics = df.describe()
    statistics.loc['std'] = df.std()
    return statistics

def create_boxplot(df):
    df.boxplot(column=['Mean_best', 'Std_best', 'Mean_f', 'Mean_acc', 'Std_scc', 'Max_acc', 'Time'], rot=45, fontsize=10)  
    plt.show()

def step(criteria):
    results = create_table(criteria)
    print(results)
    statistics = calculate_statistics(results)
    print(statistics)
    create_boxplot(results)


if __name__ == "__main__":
    criteria = {
        'function_name': None,
        'swarm_distribution': 'uniform',
        'phi_p_distribution': None,
        'phi_g_distribution': None,
        'experiment': 'total',
            }

    step(criteria)

