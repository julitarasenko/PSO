import math
import numpy as np
from cmath import pi
#from problem1 import problem1
#from problem2 import problem2
#from problem5 import problem5
#from problem6 import problem6
#from problem7 import problem7
from ftest import spheref, zakharov, rosenbrock, trid6, easom, ackley, griewank, alpine, perm, schwefel, \
    yang3, yang4, csendes, yang2, levy8, brown

def test_sets():
    # n=5
    # param_problem = {
    #     'name' : [problem1, problem2, problem5, problem6, problem7],
    #     'dim': [20, 6, 3*n, 3*n, 3*n, 20],
    #     'domain': [[-6.4, 6.5], # problem1
    #             [[0, 4], [0, 4], [0, np.pi]] + [[-4-1/4*math.floor((i-4)/3), +4+1/4*math.floor((i-4)/3)] for i in range(4,3*n+1)],
    #             # [[-4.5, 4.25], [0, 4], [0, np.pi]] + [[-4.25, 4.25] for i in range(4,3*n+1)], # problem5
    #             # [[-4.5, 4.25], [0, 4], [0, np.pi]] + [[-4.25, 4.25] for i in range(4,3*n+1)], # problem6
    #             [[0, 4], [0, 4], [0, np.pi]] + [
    #                 [-4 - 1 / 4 * math.floor((i - 4) / 3), +4 + 1 / 4 * math.floor((i - 4) / 3)] for i in
    #                 range(4, 3 * n + 1)],
    #             [[0, 4], [0, 4], [0, np.pi]] + [
    #                 [-4 - 1 / 4 * math.floor((i - 4) / 3), +4 + 1 / 4 * math.floor((i - 4) / 3)] for i in
    #                 range(4, 3 * n + 1)],
    #             [0, 2*pi]],
    #     'min': [0, -600, -600, -600, 0]
    # }

    n = 20

    # trid_xmin = []
    #
    # for i in range(6):
    #     trid_xmin.append((i + 1) * (6- i))


    # param_ftest = {
    #     'name': [ easom, ackley],
    #     'dim': [ 2, n],
    #     'domain': [ [-100, 100], [-35, 35]],
    #     'min': [ -1, 0],
    #     'x_best': [np.ones(2)*np.pi, np.zeros(n)]
    # }

    # param_ftest = {
    #     'name': [spheref, zakharov, rosenbrock,
    #              brown, easom, ackley,
    #              griewank, alpine, perm,
    #              schwefel, yang3, yang4,
    #              csendes, yang2, levy8],
    #     'dim': [n, n, n, # spheref, zakharov, rosenbrock,
    #             n, 2, n, # trid6, easom, ackley,
    #             n, n, n, # griewank, alpine, perm,
    #             n, n, n, # schwefel, yang3, yang4,
    #             n, n, n], # csendes, yang2, levy
    #     'domain': [[0, 10], [-5, 5], [-30, 30], # spheref, zakharov, rosenbrock,
    #                 [-1, 4], [-100, 100], [-35, 35], # brown not trid6 [-36, 36]n =6, easom, ackley,
    #                 [-100, 100], [-10, 10], [-n, n], # griewank, alpine, perm,
    #             [-500, 500], [-2*pi, 2*pi], [-20, 20], # schwefel, yang3, yang4,
    #             [-1, 1], [-5, 5], [-10, 10]], # csendes, yang2, levy
    #     'min': [0, 0, 0, # spheref, zakharov, rosenbrock,
    #             0, -1, 0, # brown -n*(n+4)*(n-1)/6, easom, ackley,
    #             0, 0, 0, # griewank, alpine, perm,
    #             0, 0, -1, # schwefel, yang3, yang4,
    #             0, 0, 0],#csendes, yang2, levy
    #     'x_best': [np.zeros(n), np.zeros(n), np.ones(n),# spheref, zakharov, rosenbrock
    #                np.zeros(n), np.ones(2)*np.pi, np.zeros(n), # brown, easom, ackley,
    #                np.zeros(n), np.zeros(n), list(np.arange(1, n + 1)),  # griewank, alpine, perm,
    #                np.ones(n)*420.9687, np.zeros(n), np.zeros(n),  # schwefel, yang3, yang4,
    #                np.zeros(n), np.zeros(n), np.ones(n)]  # csendes, yang2, levy8
    # }

    n = 20

    param_ftest = {
        'name': [spheref, zakharov, rosenbrock,
                 brown, easom, ackley,
                 griewank, alpine, perm,
                 schwefel, yang3, yang4,
                 csendes, yang2, levy8],
        'dim': [n, n, n, # spheref, zakharov, rosenbrock,
                n, 2, n, # trid6, easom, ackley,
                n, n, n, # griewank, alpine, perm,
                n, n, n, # schwefel, yang3, yang4,
                n, n, n], # csendes, yang2, levy
        'domain': [[0, 10], [-5, 5], [-30, 30], # spheref, zakharov, rosenbrock,
                    [-1, 4], [-100, 100], [-35, 35], # brown not trid6 [-36, 36]n =6, easom, ackley,
                    [-100, 100], [-10, 10], [-n, n], # griewank, alpine, perm,
                [-500, 500], [-2*pi, 2*pi], [-20, 20], # schwefel, yang3, yang4,
                [-1, 1], [-5, 5], [-10, 10]], # csendes, yang2, levy
        'min': [0, 0, 0, # spheref, zakharov, rosenbrock,
                0, -1, 0, # brown -n*(n+4)*(n-1)/6, easom, ackley,
                0, 0, 0, # griewank, alpine, perm,
                0, 0, -1, # schwefel, yang3, yang4,
                0, 0, 0],#csendes, yang2, levy
        'x_best': [np.zeros(n), np.zeros(n), np.ones(n),# spheref, zakharov, rosenbrock
                   np.zeros(n), np.ones(2)*np.pi, np.zeros(n), # brown, easom, ackley,
                   np.zeros(n), np.zeros(n), list(np.arange(1, n + 1)),  # griewank, alpine, perm,
                   np.ones(n)*420.9687, np.zeros(n), np.zeros(n),  # schwefel, yang3, yang4,
                   np.zeros(n), np.zeros(n), np.ones(n)],  # csendes, yang2, levy8
        'unimodal': [True, True, True, # spheref, zakharov, rosenbrock
                     True, True, False, # brown, easom, ackley,
                     False, False, False, # griewank, alpine, perm,
                     False, False, False, # schwefel, yang3, yang4,
                     False, False, False,], # csendes, yang2, levy8
        'separable': [True, False, False, # spheref, zakharov, rosenbrock
                      False, True, False, # brown, easom, ackley,
                      False, True, True, # griewank, alpine, perm,
                      False, False, False, # schwefel, yang3, yang4,
                      True, False, False] # csendes, yang2, levy8
    }

    param_problem ={}
    # n = 20
    #
    # param_ftest = {
    #     'name': [spheref, rosenbrock, griewank],
    #     'dim': [n, n, n],
    #     'domain': [[0, 10], [-5, 10], [-100, 100]],
    #     'min': [0, 0, 0]
    # }

    return param_problem, param_ftest
