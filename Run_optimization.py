from PSO_object import PSO

def run_optimization(params):
<<<<<<< Updated upstream
    dim, bound, func, minimum, iterations, x_opt, i, qmc_interval, swarm_dist, dist1, dist2, d_type = params
    pso = PSO(dim, dim, bound, func, minimum, x_opt, i, qmc_interval, swarm_dist, dist1, dist2, d_type) #PSO(no_particles, dim, bounds, func_name, expected_min)
=======
    dim, bound, func, minimum, iterations, x_opt, swarm_dist, dist1, dist2, d_type = params
    if func != 'easom':
        pso = PSO(dim, dim, bound, func, minimum, x_opt, swarm_dist, dist1, dist2, d_type) #PSO(no_particles, dim, bounds, func_name, expected_min)
    else:
        pso = PSO(20, dim, bound, func, minimum, x_opt, swarm_dist, dist1, dist2,
                  d_type)  # PSO(no_particles, dim, bounds, func_name, expected_min)
>>>>>>> Stashed changes
    res = pso.optimize(iterations) # res contains: best_position, best_score, avg_best, std_best, avg_swarm,  avg_acc, std_acc, max_acc, max_iter,
    return res[1:]
