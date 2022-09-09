import math
import numpy as np

def problem6(x):
    # x = np.array([0.217000000000000,0.0240000000000000,0.0760000000000000,0.892000000000000,0.128000000000000,0.250000000000000,0.0580000000000000,0.112000000000000,0.0620000000000000,0.0820000000000000,0.217000000000000,0.0240000000000000,0.0260000000000000,0.491000000000000,0.228000000000000,0.300000000000000,0.0580000000000000,0.112000000000000,0.0620000000000000,0.0820000000000000,0.216000000000000,0.0240000000000000,0.0760000000000000,0.216000000000000,0.216000000000000,0.216000000000000,0.0580000000000000,0.112000000000000,0.0620000000000000,0.0820000000000000])
    x = x.T
    p = np.size(x)
    if math.remainder(p,3) != 0:
        print("x passed to this function must be n dimentional array where, n is perfectly divisible by 3.")
    NP = p//3
    x = x.reshape(NP,3)
    R1 = 2.85
    R2 = 0.15
    A = 1.8308e+3
    B = 4.7118e+2
    lemda1 = 2.4799
    lemda2 = 1.7322
    lemda3 = 1.7322
    c = 1.0039e+05
    d = 1.6218e+01
    n1 = 7.8734e-01
    gama = 1.0999e-06
    h = -5.9826e-01
    E = np.zeros(NP)
    r = np.zeros((NP,NP))
    fcr = np.zeros((NP,NP))
    VRr = np.zeros((NP,NP))
    VAr = np.zeros((NP,NP))
    for i in range(NP):
        for j in range(NP):
            r[i][j] = math.sqrt((x[i] - x[j]).dot(x[i] - x[j]))
            if r[i][j]<(R1-R2):
                fcr[i][j]=1
            elif  r[i][j]>(R1+R2):
                fcr[i][j]=0
            else:
                fcr[i][j]=0.5-0.5*math.sin(math.pi/2*(r[i][j]-R1)/R2)
            VRr[i][j]=A*math.exp(-lemda1*r[i][j])
            VAr[i][j]=B*math.exp(-lemda2*r[i][j])
    for i in range(NP):
        for j in range(NP):
            if i==j:
                continue
            jeta=np.zeros((NP,NP))
            for k in range (NP):
                if i==k or j==k:
                    continue
                rd1=math.sqrt((x[i]-x[k]).dot(x[i]-x[k]))
                rd3=math.sqrt((x[k]-x[j]).dot(x[k]-x[j]))
                rd2=math.sqrt((x[i]-x[j]).dot(x[i]-x[j]))
                if (rd1 != 0 and rd2 != 0):
                    ctheta_ijk=(rd1**2+rd2**2-rd3**3)/(2*rd1*rd2)
                if (d**2+(h-ctheta_ijk)**2 != 0):
                    G_th_ijk =1+(c**2)/(d**2)-(c**2)/(d**2+(h-ctheta_ijk)**2)
                jeta[i][j]+=fcr[i][k]*G_th_ijk*math.exp(lemda3**3*(r[i][j]-r[i][k])**3)

            Bij=(1+(gama*jeta[i][j])**n1)**(-0.5/n1)
            E[i]=E[i]+fcr[i][j]*(VRr[i][j]-Bij*VAr[i][j])/2
    f=sum(E)
    return f

# f = problem6()
# print(f)