import math
import numpy as np

def problem1():
    x=[0.217000000000000,0.0240000000000000,0.0760000000000000,0.892000000000000,0.128000000000000,0.250000000000000]
    d = np.size(x)
    if d<6:
        print('dimension-size should be six.')
    if d>6:
        print('dimension-size is more than 6.')
        print('function has been evaluated on first six dimensions.')
    theta=2*math.pi/100
    f=0
    for t in range (100):
        y_t=x[0]*math.sin(x[1]*t*theta+x[2]*math.sin(x[3]*t*theta+x[4]*math.sin(x[5]*t*theta)))
        y_0_t=1*math.sin(5*t*theta-1.5*math.sin(4.8*t*theta+2*math.sin(4.9*t*theta)))
        f=f+(y_t-y_0_t)**2
    return f

f = problem1()
print(f)