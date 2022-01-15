import math

def branin(xx):
    a = 1
    b = 5.1/(4*math.pi**2)
    c = 5/math.pi
    r = 6
    s = 10
    t = 1/(8*math.pi)
    x1 = xx[0]
    x2 = xx[1]
    if (len(xx) < 7):
        t = 1 / (8*math.pi)
    if (len(xx) < 6):
        s = 10
    if (len(xx) < 5):
        r = 6
    if (len(xx) < 4):
        c = 5/math.pi
    if (len(xx) < 3):
        b = 5.1 / (4*math.pi**2)
    if (len(xx) < 2):
        a = 1
    term1 = a * (x2 - b*x1**2 + c*x1 - r)**2
    term2 = s*(1-t)*math.cos(x1)
    y = term1 + term2 + s
    return y

def easom(xx):
    x1 = xx[0]
    x2 = xx[1]
    fact1 = -math.cos(x1)*math.cos(x2)
    fact2 = math.exp(-(x1-math.pi)**2-(x2-math.pi)**2)
    y = fact1*fact2
    return y

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

def rastr(xx):
    d = len(xx)
    sum = 0
    for ii in range(d):
        xi = xx[ii]
        sum = sum + (xi**2 - 10*math.cos(2*math.pi*xi))
    y = 10*d + sum
    return y

def spheref(xx):
    d = len(xx)
    sum = 0
    for ii in range(d):
        xi = xx[ii]
        sum = sum + xi**2
    y = sum
    return y

def stybtang(xx):
    d = len(xx)
    sum = 0
    for ii in range(d):
        xi = xx[ii]
        new = xi**4 - 16*xi**2 + 5*xi
        sum = sum + new
    y = sum/2
    return y

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