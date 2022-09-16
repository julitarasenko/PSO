import numpy as np

#Funkcje unimodalne: 
#20. Sphere function (separable)
def spheref(xx):
    return np.sum(np.square(xx), axis=1)

#24. Zakharov function (nonseparable) 
def zakharov(xx):
    d = len(xx.T)
    arr = []
    for ii in range(d):
        arr.append(ii)
    sum1 = np.sum(np.square(xx), axis = 1)
    sum2 = np.sum(arr * xx * 0.5, axis = 1)
    return sum1 + sum2**2 + sum2**4

#Rosenbrock function (nonseparable) global 
def rosenbrock(xx):
    arr1 = xx[:, 0:len(xx.T) - 1]
    arr2 = xx[:, 1:]
    return np.sum(100*np.square(arr2 - np.square(arr1)) + np.square(arr1 - 1), axis = 1)

#Modified Rosenbrock function (nonseparable) 
def modified_rosenbrock(xx):
    xx = 15 * xx - 5
    arr1 = xx[:, 0:len(xx.T) - 1]
    arr2 = xx[:, 1:]
    return (np.sum(100*np.square(arr2 - np.square(arr1)) + np.square(arr1 - 1), axis = 1) - 382700) / 375500

#Easom
def easom(xx):
    x1 = xx[:, 0]
    x2 = xx[:, 1]
    fact1 = - np.cos(x1) * np.cos(x2)
    fact2 = np.exp( -(x1 - np.pi)**2 - (x2 - np.pi)**2)
    return fact1 * fact2

#Multimodal: 
# 1. Ackley function (nonseparable) 
def ackley(xx):
    d = len(xx.T)
    c = 2 * np.pi
    b = 0.2
    a = 20
    sum1 = np.sum(np.square(xx), axis=1)
    sum2 = np.sum(np.cos(c * xx), axis=1)
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

# 34. Griewank function (nonseparable)
def griewank(xx):
    d = len(xx.T)
    arr = []
    for ii in range(d):
        arr.append(ii + 1)
    sum = np.sum(np.square(xx) / 4000, axis = 1)
    prod = np.prod(np.cos(xx / np.sqrt(arr)), axis = 1)
    return sum - prod + 1 

# 2. Alpine function (separable) 
def alpine(xx):
    return np.sum(xx * np.sin(xx) + 0.1*xx, axis = 1)

# 76. Perm function (separable) 
def perm(xx):
    b = 10
    d = len(xx.T)
    arr = []
    arrb = []
    for ii in range(d):
        arr.append(ii + 1)
        arrb.append(ii + 1 + b)
    outer = 0
    for kk in range(d):
        inner = np.sum(arrb * ((xx / arr)**(kk + 1) - 1), axis = 1)
        outer += inner**2
    return outer

# 86. Schwefel function (nonseparable) 
def schwefel(xx):
    d = len(xx.T)
    sum = np.sum(xx * np.sin(np.sqrt(abs(xx))), axis = 1)
    return 418.9829 * d - sum

# 112. Yang 3 function (nonseparable) 
def yang3(xx):
    sum1 = np.sum(abs(xx), axis = 1)
    sum2 = np.sum(np.sin(xx**2), axis = 1)
    return sum1 * np.exp(-sum2)

# 113. Yang 4 function (nonseparable) 
def yang4(xx):
    b = 15
    m = 5
    sum1 = np.sum((xx / b)**(2 * m), axis = 1)
    sum2 = np.sum(xx**2, axis = 1)
    mult = np.prod(np.cos(xx)**2, axis = 1)
    return np.exp(-sum1) - 2 * np.exp(-sum2) * mult

# 21. Csendes function (separable) 
def csendes(xx):
    return np.sum(xx**6 * (2 + np.sin(1 / xx)), axis = 1)

# 111. Yang 2 function (separable) 
def yang2(xx):
    d = len(xx.T)
    rand = np.random.uniform(0,1,d)
    arr = []
    for ii in range(d):
        arr.append(ii + 1)
    return np.sum(rand * abs(xx)**arr, axis = 1)

# 55. Levy 8 function (nonseparable) 
def levy(xx):
    d = len(xx.T)
    w = 1 + (xx - 1) / 4
    term1 = (np.sin(np.pi*w[:, 0]))**2
    term3 = (w[:, d-1] - 1)**2 * (1 + (np.sin(2 * np.pi * xx[: , d-1]))**2)
    sum = np.sum((w[:, 0:d-2] - 1)**2 * (1 + 10 * (np.sin(np.pi * w[:, 0:d-2] + 1))**2), axis = 1)
    return term1 + sum + term3