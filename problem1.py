import math
import numpy as np


# I have change the function to accpet array not only single vector
def problem1(x):
    x = x.T
    #x=[0.217000000000000,0.0240000000000000,0.0760000000000000,0.892000000000000,0.128000000000000,0.250000000000000]
    if len(x) == np.size(x):
        d1 = np.size(x)
    else:
        d1, d2 = x.shape

    if d1<6:
        print('dimension-size should be six.')
    if d1>6:
        print('dimension-size is more than 6.')
        print('function has been evaluated on first six dimensions.')
    theta=2*math.pi/100
    f=0
    for t in range(100):
        tt= t * theta
        y_t = x[0] * np.sin(x[1] * tt + x[2] * np.sin(x[3] * tt + x[4] * np.sin(x[5] * tt)))
        y_0_t = np.sin(5 * tt - 1.5 * np.sin(4.8 * tt + 2 * np.sin(4.9 * tt)))
        f = f + (y_t - y_0_t)**2
    return f

# f = problem1()
# print(f)