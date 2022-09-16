import math
import numpy as np

def problem7(xx):
    # x = np.array([0.217000000000000,0.0240000000000000,0.0760000000000000,0.892000000000000,0.128000000000000,0.250000000000000,0.0580000000000000,0.112000000000000,0.0620000000000000,0.0820000000000000,0.0350000000000000,0.0900000000000000,0.0320000000000000,0.0950000000000000,0.0220000000000000,0.175000000000000,0.0320000000000000,0.0870000000000000,0.0350000000000000,0.0240000000000000])
    d = len(xx.T)
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

# f = problem7()
# print(f)