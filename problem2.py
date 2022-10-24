import math
import numpy as np

def problem2(x):
    d1, d = x.shape
    # if d1%3 != 0:
    #     print('x passed to this function must be n dimentional array where, n is perfectly divisible by 3.')
    n = d//3
    a = np.ones((n,n))
    b = 2 * a
    x_= x.reshape(d1,n,3)
    # calculating cartesian distance between n atoms
    r = np.sqrt(np.sum((x_[:,range(n-1),:]-x_[:,range(1,n),:])**2,axis=2))
    v = np.sum(1/r**12 - 2/r**6, axis=1)
    return v