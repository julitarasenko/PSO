import math
import numpy as np

# Change to taking only 6 first elements from the pool
def problem1(x):
    x = x.T
    # if len(x) == np.size(x):
    #     d1 = np.size(x)
    # else:
    #     d1, d2 = x.shape
    #
    # if d1<6:
    #     print('dimension-size should be six.')
    # if d1>6:
    #     print('dimension-size is more than 6.')
    #     print('function has been evaluated on first six dimensions.')
    theta=2*math.pi/100
    f = 0
    for t in range(100):
        tt = t * theta
        y_t = x[0] * np.sin(x[1] * tt + x[2] * np.sin(x[3] * tt + x[4] * np.sin(x[5] * tt)))
        y_0_t = np.sin(5 * tt - 1.5 * np.sin(4.8 * tt + 2 * np.sin(4.9 * tt)))
        f = f + (y_t - y_0_t)**2
    return f