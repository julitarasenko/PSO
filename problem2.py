import math
import numpy as np

def problem2(x):
    d1 = np.size(x)
    if d1%3 != 0:
        print('x passed to this function must be n dimentional array where, n is perfectly divisible by 3.')
    n = d1//3
    v = 0
    a = np.ones((n,n))
    b = a.dot(2)
    r = np.zeros((n-1,n))
    for i in range (n-1):
        for j in range(i+1, n):
            r[i][j] = np.sqrt((x[i] - x[j]).dot(x[i] - x[j]))
            if (r[i][j] != 0):
                v = v + (a[i][j]/r[i][j]**12 - b[i][j]/r[i][j]**6)
    f = v
    return f