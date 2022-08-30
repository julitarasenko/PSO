"""PSO function
    :parameter
        d <- dimentions
        swarm_size <- number of particles
        phi_p <- personal weight
        phi_g <- global weight
        setup <- dictionary with keys:
            swarm:  list of tuples with random swarm initialization
            phi: list of tuples with random phi_p and phi_g initialization
            v: velocity initialization
"""

def pso(d,swarm_size,domain,setup): # Hyper-parameter of the algorithm
    """A loop for making all initialization"""
    print(setup[0], setup[1])

    w = setup[0][1](size=1)
    phi_p = setup[0][2][0](size=d)
    phi_g = setup[0][3][0](size=d)

    print(w, phi_p, phi_g)

    #
    # # Create particles inside the range
    X = (domain[1]-domain[0]) * (setup[0][0][0](size=(d, swarm_size)) - domain[0])
    V = numpy.zeros(d, swarm_size) # lub losowe wartości z przedziału 1/2domain