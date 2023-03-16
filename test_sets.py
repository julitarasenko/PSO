import math
import numpy as np
from cmath import pi
from problem1 import problem1
from problem2 import problem2
from problem5 import problem5
from problem6 import problem6
from problem7 import problem7
from ftest import spheref, zakharov, rosenbrock, trid6, easom, ackley, griewank, alpine, perm, schwefel, \
    yang3, yang4, csendes, yang2, levy8

def test_sets():
    n=5
    param_problem = {
        'name' : [problem1, problem2, problem5, problem6, problem7],
        'dim': [20, 6, 3*n, 3*n, 3*n, 20],
        'domain': [[-6.4, 6.5], # problem1
                [[0, 4], [0, 4], [0, np.pi]] + [[-4-1/4*math.floor((i-4)/3), +4+1/4*math.floor((i-4)/3)] for i in range(4,3*n+1)],
                # [[-4.5, 4.25], [0, 4], [0, np.pi]] + [[-4.25, 4.25] for i in range(4,3*n+1)], # problem5
                # [[-4.5, 4.25], [0, 4], [0, np.pi]] + [[-4.25, 4.25] for i in range(4,3*n+1)], # problem6
                [[0, 4], [0, 4], [0, np.pi]] + [
                    [-4 - 1 / 4 * math.floor((i - 4) / 3), +4 + 1 / 4 * math.floor((i - 4) / 3)] for i in
                    range(4, 3 * n + 1)],
                [[0, 4], [0, 4], [0, np.pi]] + [
                    [-4 - 1 / 4 * math.floor((i - 4) / 3), +4 + 1 / 4 * math.floor((i - 4) / 3)] for i in
                    range(4, 3 * n + 1)],
                [0, 2*pi]],
        'min': [0, -600, -600, -600, 0]
    }

    n = 20

    param_ftest = {
        'name': [spheref, zakharov, rosenbrock, trid6, easom, ackley, griewank, alpine, perm, schwefel, yang3,
                yang4, csendes, yang2, levy8],
        'dim': [n, n, n, n, 2, n, n, n, n, n, n, n, n, n, n],
        'domain': [[0, 10], [-5, 5], [-5, 10], # spheref, zakharov, rosenbrock,
                    [-36, 36], [-100, 100], [-35, 35], # trid6, easom, ackley,
                    [-100, 100], [-10, 10], [-n, n], # griewank, alpine, perm,
                [-500, 500], [-2*pi, 2*pi], [-20, 20], # schwefel, yang3, yang4,
                [-1, 1], [-5, 5], [-10, 10]], # csendes, yang2, levy
        'min': [0, 0, 0, -50, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0]
    }

    return param_problem, param_ftest