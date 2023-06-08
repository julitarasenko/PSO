import pandas as pd
import math
from Run_optimization import run_optimization

def HalvingSHA(generator_set, func, dim, bound, min_value, x_opt, d_type):
    #minimum resources
    r = 5

    # maximum resources
    R = len(generator_set['swarm']) * len(generator_set['phi_p']) * len(generator_set['phi_g'])

    # print("R: ", R)
    base = 2

    sMax = math.ceil(math.log(R/r, base))
    s = 0

    # Generating table of setups from all combinations, where the list contains
    # 0 - swarm
    # 1 - omega
    # 2 - phi_p
    # 3 - phi_g
    setup = [] # store all combinations of 4 params combinations
    for i1 in generator_set['swarm']:
        for i2 in generator_set['phi_p']:
            for i3 in generator_set['phi_g']:
                    setup.append([i1, i2, i3])

    df_result = pd.DataFrame(columns=["setIdx", "Swarm", "phiP", "phiG",
                                      "MaxIter", "BestScore", "Mean_best",
                                      "Std_best", "Mean_f", "Mean_acc", "Std_acc",
                                      "Max_acc", "Iteration", "Time",])
    df_result.to_csv(f"resultSHA-{str(func).split(' ')[1]}.csv", index=False)

    # The halving algorithm
    index = list(range(R))
    for i in range(0,(sMax-s)):
        ni = math.floor(R*math.pow(2, -i)) #number of setups for the iteration
        ri = r * math.pow(2, (i+s)) #number of resources in the iteration or swarm size?

        for j in index[:ni]:
            max_iter = int(ri*5) # might be connected with the algorithm as the resource
            
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
            df_new_row.to_csv(f"resultSHA-{str(func).split(' ')[1]}.csv", mode='a', header=False, index=False)  
        
        if df_result['Mean_best'].max() == min_value:
            Mean_best = 0
        else:
            Mean_best = (df_result['Mean_best'] - min_value) / (df_result['Mean_best'].max() - min_value)

        if df_result['Mean_acc'].max() == 0:
            Mean_acc = 0
        else:
            Mean_acc = df_result['Mean_acc'] / df_result['Mean_acc'].max()

        if df_result['Iteration'].max() == 0:
            Iteration = 0
        else:
            Iteration = df_result['Iteration'] / df_result['Iteration'].max()

        df_result['Criteria'] = Mean_best - Mean_acc + Iteration
        df_result['BestRank'] = df_result['Criteria'].rank(ascending=False, pct=True)
        # Sort by rank and store in index vector
        df_result.sort_values(by='BestRank', ascending=False, inplace=True)
        index = df_result['setIdx'].values
        df_result = pd.DataFrame(columns=["setIdx", "Swarm", "phiP", "phiG",
                                      "MaxIter", "BestScore", "Mean_best",
                                      "Std_best", "Mean_f", "Mean_acc", "Std_acc",
                                      "Max_acc", "Iteration", "Time",])                                          


    
    


