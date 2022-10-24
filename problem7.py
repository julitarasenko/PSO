import math
from turtle import shape
import numpy as np

def problem7(xx):
    d1, d = xx.shape
    m = 2*d-1
    hsum = np.zeros((d1, 2*m +1))
    for kk in range(1, m+1):
        if kk % 2 != 0:
            i = int((kk + 1) / 2)
            hsum[:, kk] = 0
            for j in range(i, d+1):  # fi(2i-1)X
                summ = np.zeros(d1)
                for i1 in range(abs(2 * i - j - 1)+1, j+1):
                    summ += xx[:, i1-1]
                hsum[:, kk] += np.cos(summ)
        else:
            i = int(kk/2)
            hsum[:,kk] = 0
            for j in range(i+1, d+1):    #fi(2i)X
                summ = 0
                for i1 in range(abs(2 * i - j)+1, j+1):
                    summ += xx[:,i1-1]
                hsum[:,kk] += np.cos(summ)
            hsum[:,kk] += 0.5

    hsum[:,2*d:] = -hsum[:,1:2*d]
    return np.max(hsum, axis=1)