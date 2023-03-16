import numpy as np
import time

#Funkcje unimodalne: 
#20. Sphere function (separable)
def spheref(xx):
    return np.sum(np.square(xx), axis=1)

#24. Zakharov function (nonseparable) 
def zakharov(xx):
    d1, d = xx.shape
    arr = np.linspace(1, d, d)
    sum1 = np.sum(xx**2, axis=1)
    sum2 = np.sum(arr * xx, axis=1) * 0.5
    return sum1 + sum2**2 + sum2**4

#Rosenbrock function (nonseparable) global 
def rosenbrock(xx):
    d1, d = xx.shape
    arr1 = xx[:, 0 : d - 1]
    arr2 = xx[:, 1:]
    return np.sum(100*(arr2 - arr1**2)**2 + (1-arr1)**2, axis=1)

#Trid 6 function (nonseparable)
def trid6(xx):
    d1, d = xx.shape
    arr1 = xx[:, 0: d - 1]
    arr2 = xx[:, 1:]
    return (np.sum((xx - 1)**2, axis=1) - np.sum(arr2*arr1,axis=1))

#Easom
def easom(xx):
    fact1 = np.prod(np.cos(xx), axis=1)
    fact2 = np.exp(-np.sum((xx - np.pi)**2, axis=1))
    return (-fact1 * fact2)

#Multimodal: 
# 1. Ackley function (nonseparable) 
def ackley(xx):
    d1, d = xx.shape
    c = 2 * np.pi
    a = 20
    sum1 = np.sum(xx**2, axis=1)
    sum2 = np.sum(np.cos(c * xx), axis=1)
    term1 = -a * np.exp(-0.02 * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

# 34. Griewank function (nonseparable)
def griewank(xx):
    d1, d = xx.shape
    arr = np.linspace(1, d, d)
    sum = np.sum(xx**2 / 4000, axis=1)
    prod = np.prod(np.cos(xx / np.sqrt(arr)), axis=1)
    return sum - prod + 1 

# 2. Alpine function (separable) 
def alpine(xx):
    return np.sum(np.abs(xx * np.sin(xx) + 0.1*xx), axis = 1)

# 76. Perm function (separable) 
def perm(xx):
    b = 0.5
    d1, d = xx.shape
    arr = np.linspace(1, d, d)
    arrb = np.linspace(1 + b, d + b, d)
    outer = 0
    for kk in range(1,d+1):
        inner = np.sum(arrb * ((xx / arr)**kk - 1), axis = 1)
        outer += inner**2
    return outer

# 86. Schwefel function (nonseparable) 
def schwefel(xx):
    d1, d = xx.shape
    sum = np.sum(xx * np.sin(np.sqrt(abs(xx))), axis=1)
    return 418.9829 * d - sum

# 112. Yang 3 function (nonseparable) 
def yang3(xx):
    sum1 = np.sum(abs(xx), axis=1)
    sum2 = np.sum(np.sin(xx**2), axis=1)
    return sum1 * np.exp(-sum2)

# 113. Yang 4 function (nonseparable) 
def yang4(xx):
    b = 15
    m = 5
    sum1 = np.sum((xx / b)**(2 * m), axis=1)
    sum2 = np.sum(xx**2, axis=1)
    mult = np.prod(np.cos(xx)**2, axis=1)
    return np.exp(-sum1) - 2 * np.exp(-sum2) * mult

# 21. Csendes function (separable) 
def csendes(xx):
    return np.sum(xx**6 * (2 + np.sin(1 / xx)), axis=1)

# 111. Yang 2 function (separable) 
def yang2(xx):
    d1, d = xx.shape
    rand = np.random.uniform(0,1,d)
    arr = np.linspace(1, d, d)
    return np.sum(rand * abs(xx)**arr, axis=1)

# 55. Levy 8 function (nonseparable) 
def levy8(xx):
    d1, d = xx.shape
    w = 1 + (xx - 1) / 4
    term1 = (np.sin(np.pi * w[:, 0]))**2
    term3 = (w[:, d-1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[:, d-1]))**2)
    sum = np.sum((w[:, 0:d-1] - 1)**2 * (1 + 10 * (np.sin(np.pi * w[:, 0:d-1] + 1))**2), axis = 1)
    return term1 + sum + term3