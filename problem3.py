from array import array
import math
import numpy as np
import numpy.matlib
import scipy.integrate as integrate

def diffsolv(t,x,u):
    # load c_bifunc_data;% c(i,j) is saved here.
    ml=np.array([1, u, u.dot(u), u.dot(u.dot(u))])
    mlt=np.matlib.repmat(ml,10,1)
    arr=t.dot(mlt)
    k=arr.sum(axis=1)
    dy = np.zeros((7,1))    # a column vector
    print(dy)
    print(x)
    print(k)
    dy[0] = -k[0]*x[0]
    dy[1] = k[0]*x[0]-(k[1]+k[2])*x[1]+k[3]*x[4]
    dy[2] = k[1]*x[1]
    dy[3] = -k[5]*x[3]+k[4]*x[4]
    dy[4] = k[2]*x[1]+k[5]*x[3]-(k[3]+k[4]+k[7]+k[8])*x[4]+k[6]*x[5]+k[9]*x[6]
    dy[5] = k[7]*x[4]-k[6]*x[5]
    dy[6] = k[8]*x[4]-k[9]*x[6]
    return dy

def problem3():
    x=np.array([0.045])
    tol=1.0e-01 # tol is a matter of concern. decreasing it make algo fast. 
    tspan=np.array([0, 0.78]) # check for the range
    yo =np.array([1, 0, 0, 0, 0, 0, 0])
    u=x #u should be passed here.
    t=np.arange(0,10)
    t=t/10
    y=np.arange(0,10)
    y=y/10
    T, Y = integrate.ode(diffsolv(t,y,u)).set_integrator('dopri5').set_initial_value(tspan, yo, 'rtol', 'atol')
    w=np.size(Y)
    f=Y(w[0],w[1])*1e+003
    return f

f = problem3()
print(f)