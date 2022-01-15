import math

def analiz(g_best):
    suma = sum(g_best)
    srednie = suma/len(g_best)
    a = 0
    for i in range(len(g_best)):
        a = a + (g_best[i] - srednie)**2
    a = a / srednie
    sigma = math.sqrt(abs(a))
    lepszy = min(g_best)
    gorszy = max(g_best)
    return srednie,sigma,lepszy,gorszy