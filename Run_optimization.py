from PSO_object import PSO
from DE_object import DE

def run_optimization(params):
    dim, bound, func, minimum, iterations, x_opt, swarm_dist, dist1, dist2, d_type = params[0]
    pso = PSO(dim, dim, bound, func, minimum, x_opt, swarm_dist, dist1, dist2, d_type) #PSO(no_particles, dim, bounds, func_name, expected_min)
    res = pso.optimize(iterations) # res contains: best_position, best_score, avg_best, std_best, avg_swarm,  avg_acc, std_acc, max_acc, max_iter,
    return res[1:]
