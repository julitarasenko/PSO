import math
from turtle import shape
import numpy as np

def problem7(xx):
    d1, d = xx.shape
    hsum = np.zeros(2*(2*d-1))
    var = 2*d-1
    f = []
    for x in xx:
        for kk in range(2*var):
            if np.remainder(kk,2)!=0:
                i = (kk+1)/2
                hsum[kk] = 0
                for j in range(math.floor(i-1), d):   #fi(2i-1)X
                    summ = 0
                    for i1 in range (math.floor(abs(2*i-j-1)), j):
                        summ += x[i1]
                    hsum[kk] += np.cos(summ)
            else:
                i = kk/2
                hsum[kk] = 0
                for j in range(math.floor(i), d):    #fi(2i)X
                    summ = 0
                    for i1 in range(math.floor(abs(2*i-j)), j):
                        summ += x[i1]
                    hsum[kk] += np.cos(summ)

                hsum[kk] += 0.5
        f.append(max(hsum[1:2*(2*d-1)]))
    f = np.array(f)
    return f