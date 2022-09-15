import math
import numpy as np

#Funkcje unimodalne: 
#20. Sphere function (separable)
def spheref(xx):

    return np.sum(np.square(xx), axis=1)
    # d = len(xx)
    # sum = 0
    # for ii in range(d):
    #     xi = xx[ii]
    #     sum = sum + xi**2
    # y = sum
    # return y

#24. Zakharov function (nonseparable) 
def zakharov(xx):
    d = len(xx)
    sum1 = 0
    sum2 = 0
    for ii in range(d):
        xi = xx[ii]
        sum1 = sum1 + xi**2
        sum2 = sum2 + 0.5*ii*xi
    y = sum1 + sum2**2 + sum2**4
    return y

#Rosenbrock function (nonseparable) global 
def rosenbrock(xx):
    d = len(xx)
    sum = 0
    for ii in range(d):
        sum += 100*(xx[ii+1]-xx[ii]**2)**2+(xx[ii]-1)**2
    y = sum
    return y

#Modified Rosenbrock function (nonseparable) 
def modified_rosenbrock(xx):
    x1 = xx[0]
    x2 = xx[1]
    y = 74 + 100*(x2-x1**2)**2 + (1-x1)**2 - 400*math.exp(-(x1+1)**2 + (x2+1)**2/0.1)
    return y

#Easom
def easom(xx):
    x1 = xx[0]
    x2 = xx[1]
    fact1 = -math.cos(x1)*math.cos(x2)
    fact2 = math.exp(-(x1-math.pi)**2-(x2-math.pi)**2)
    y = fact1*fact2
    return y

#Multimodal: 
# 1. Ackley function (nonseparable) 
def ackley(xx):
    d = len(xx)
    c = 2*math.pi
    b = 0.2
    a = 20
    sum1 = 0
    sum2 = 0
    for ii in range(d):
        xi = xx(ii)
        sum1 = sum1 + xi**2
        sum2 = sum2 + math.cos(c*xi)
    term1 = -a * math.exp(-b*math.sqrt(sum1/d))
    term2 = -math.exp(sum2/d)
    y = term1 + term2 + a + math.exp(1)
    return y

# 34. Griewank function (nonseparable)
def griewank(xx):
    d = len(xx)
    sum = 0
    prod = 1
    for ii in range(d):
        xi = xx[ii]
        sum = sum + xi**2/4000
        prod = prod * math.cos(xi/math.sqrt(ii+1))
    y = sum - prod + 1
    return y 

# 2. Alpine function (separable) 
def alpine(xx):
    d = len(xx)
    sum = 0
    for ii in range(d):
        xi = xx[ii]
        sum += xi*math.sin(xi) + 0.1*xi
    y = sum
    return y 

# 76. Perm function (separable) 
def perm(xx):
    b = 10
    d = len(xx)
    outer = 0
    for kk in range(d):
        inner = 0
        for ii in range(d):
            xi = xx[ii]
            inner += (ii+b)*((xi/ii)**kk-1**kk)
        outer += inner**2
    y = outer
    return y

# 86. Schwefel function (nonseparable) 
def schwefel(xx):
    d = len(xx)
    sum = 0
    for ii in range(d):
        xi = xx(ii)
        sum = sum + xi*math.sin(math.sqrt(abs(xi)))
    y = 418.9829*d - sum
    return y

# 112. Yang 3 function (nonseparable) 
def yang3(xx):
    d = len(xx)
    sum1 = 0
    sum2 = 0
    for ii in range(d):
        sum1 += abs(xx[ii])
        sum2 += math.sin(xx[ii]**2)
    y = sum1*math.exp(-sum2)
    return y

# 113. Yang 4 function (nonseparable) 
def yang4(xx):
    d = len(xx)
    b = 15
    m = 5
    sum1 = 0
    sum2 = 0
    mult = 1
    for ii in range(d):
        xi = xx[ii]
        sum1 += (xi/b)**(2*m)
        sum2 += xi**2
        mult *= math.cos(xi)**2
    y = math.exp(-sum1) - 2*math.exp(-sum2)*mult
    return y

# 21. Csendes function (separable) 
def csendes(xx):
    d = len(xx)
    sum = 0
    for ii in range(d):
        xi = xx[ii]
        sum += xi**6 * (2 + math.sin(1/xi))
    y = sum
    return y 

# 111. Yang 2 function (separable) 
def yang2(xx):
    d = len(xx)
    sum = 0
    for ii in range(d):
        xi = xx[ii]
        rand = np.random.uniform(0,1,d)
        sum += rand[ii]*abs(xi)**ii
    y = sum
    return y

# 55. Levy 8 function (nonseparable) 
def levy(xx):
    d = len(xx)
    w = np.zeros(d)
    for ii in range(d):
        w[ii] = 1 + (xx[ii] - 1)/4
    term1 = (math.sin(math.pi*w[1]))**2
    term3 = (w(d)-1)**2 * (1+(math.sin(2*math.pi*w(d)))**2)
    sum = 0
    for ii in range(d-1):
        wi = w[ii]
        new = (wi-1)**2 * (1+10*(math.sin(math.pi*wi+1))**2)
        sum = sum + new
    y = term1 + sum + term3
    return y
