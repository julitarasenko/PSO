import math
import numpy as np

def problem5(x):
    d1 = np.size(x)
    if d1%3 != 0:
        print("x passed to this function must be n dimentional array where, n is perfectly divisible by 3.")
    NP = d1//3
    R1 = 3.0
    R2 = 0.2
    A = 3.2647e+3
    B = 9.5373e+1
    lemda1 = 3.2394
    lemda2 = 1.3258
    lemda3 = 1.3258
    c = 4.8381
    d = 2.0417
    n1 = 22.956
    gama = 0.33675
    h = 0
    E = np.zeros(NP)
    r = np.zeros((NP,NP))
    fcr = np.zeros((NP,NP))
    VRr = np.zeros((NP,NP))
    VAr = np.zeros((NP,NP))
    for i in range(NP):
        for j in range(NP):
            r[i][j] = np.sqrt((x[i] - x[j]).dot(x[i] - x[j]))
            if r[i][j]<(R1-R2):
                fcr[i][j]=1
            elif  r[i][j]>(R1+R2):
                fcr[i][j]=0
            else:
                fcr[i][j]=0.5-0.5*np.sin(np.pi/2*(r[i][j]-R1)/R2)
            VRr[i][j]=A*np.exp(-lemda1*r[i][j])
            VAr[i][j]=B*np.exp(-lemda2*r[i][j])
    for i in range(NP):
        for j in range(NP):
            if i==j:
                continue
            jeta = np.zeros((NP,NP))
            for k in range (NP):
                if i==k or j==k:
                    continue
                rd1=np.sqrt((x[i]-x[k]).dot(x[i]-x[k]))
                rd3=np.sqrt((x[k]-x[j]).dot(x[k]-x[j]))
                rd2=np.sqrt((x[i]-x[j]).dot(x[i]-x[j]))
                if (rd1 != 0 and rd2 != 0):
                    ctheta_ijk=(rd1**2+rd2**2-rd3**3)/(2*rd1*rd2)
                else: 
                    ctheta_ijk = 0
                if (d**2+(h-ctheta_ijk)**2 != 0):
                    G_th_ijk =1+(c**2)/(d**2)-(c**2)/(d**2+(h-ctheta_ijk)**2)
                else:
                    G_th_ijk = 1
                jeta[i][j]+=fcr[i][k]*G_th_ijk*np.exp(lemda3**3*(r[i][j]-r[i][k])**3)

            Bij=(1+(gama*jeta[i][j])**n1)**(-0.5/n1)
            E[i]=E[i]+fcr[i][j]*(VRr[i][j]-Bij*VAr[i][j])/2
    f=sum(E)
    return f