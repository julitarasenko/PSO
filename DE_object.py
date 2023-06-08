"""Differential evolution search of the two-dimensional sphere objective function
    :parameter
        d <- dimentions
        swarm_size <- number of particles
        max_iter <- define number of max_iterations 
        domain <- define lower and upper domain for every dimension
        F <- define scale factor for mutation
        cr <- define crossover rate for recombination
        sets <- dictionary with keys:
            swarm:  list of tuples with random swarm initialization
            F, cr: list of tuples with random F and cr initialization
        fitness <- problem to be solved
"""

import numpy as np
from numpy.random import rand
from numpy.random import choice
from numpy import clip
from scipy import stats as ss
import time
import math
from matplotlib import pyplot

 
class DE:
    def __init__(self, n_particles, dim, bounds, test_func, expected_min, expected_x,
                 swarm_random=None, F_random=None, cr_random=None, d_type=np.float64):
        self.n_particles = n_particles
        self.dim = dim
        self.bounds = bounds
        self.func = test_func
        self.expected_min = expected_min
        self.expected_x = expected_x
        self.swarm_random = swarm_random
        self.F_random = F_random
        self.cr_random = cr_random
        self.d_type = d_type
        self.positions = self.initialize_positions()
        self.best_positions = self.positions.copy()
        self.best_scores = self.func(self.positions)
        self.global_best_position = None
        self.update_global_best()
        self.best_scores_history = []
        self.average_scores_history = []
        self.accuracy_history = []

    def initialize_positions(self):
        np.random.seed()
        if self.swarm_random is None:
            return np.random.uniform(self.bounds[0], self.bounds[1], (self.n_particles, self.dim)).astype(self.d_type)
        elif isinstance(self.swarm_random, type):
            x_ = self.swarm_random(d=self.dim).random(n=self.n_particles).astype(self.d_type)
            return ss.qmc.scale(x_, l_bounds=[self.bounds[0]]*self.dim, u_bounds=[self.bounds[1]]*self.dim) #Transformation to a given range
        else:
            x_ = self.swarm_random.rvs(size=(self.n_particles, self.dim)).astype(self.d_type)
            # Transformation to a given domain
            old_min, old_max = x_.min(), x_.max()
            return ((x_ - old_min) / (old_max - old_min)) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

    def initialize_velocities(self):
        return np.random.standard_normal((self.n_particles, self.dim)).astype(self.d_type) #normal
        # np.np.random.uniform(-1, 1, (n_particles, dim)).astype(self.d_type) # uniform

    def generate_F(self, size=1):
        np.random.seed()
        if self.F_random is None:
            return np.random.rand(size, self.dim)
        else:
            return self.F_random.rvs(size=size).astype(self.d_type)[0]

    def generate_cr(self, size=1):
        np.random.seed()
        if self.cr_random is None:
            return np.random.rand(size, self.dim)
        else:
            return self.cr_random.rvs(size=size).astype(self.d_type)[0]
    
   
     # define mutation operation
    def mutation(self, x):
        return x[0] + self.generate_F() * (x[1] - x[2])

    # define boundary check operation
    def check_domain(self, mutated):
        mutated_bound = clip(mutated, self.bounds[0], self.bounds[1])
        return mutated_bound
    
    
    # define crossover operation
    def crossover(self, mutated, index_positions):
        # generate a uniform random value for every dimension
        p = rand(self.dim)
        # generate trial vector by binomial crossover
        trial = [mutated[i] if p[i] < self.generate_cr() else self.best_positions[index_positions][i] for i in range(self.dim)]
        return trial
    
    def update_global_best(self):
        best_particle_idx = np.argmin(self.best_scores)
        if self.global_best_position is None or self.best_scores[best_particle_idx] < self.func(
                np.array([self.global_best_position])):
            self.global_best_position = self.best_positions[best_particle_idx].copy()

    def record_scores(self):
        self.best_scores_history.append(np.min(self.best_scores))
        self.average_scores_history.append(np.mean(self.best_scores))
        self.accuracy_history.append(self.accuracy())

    def accuracy(self):
        acc = 1 - math.dist(self.expected_x, self.global_best_position) / math.dist([self.bounds[0]] * self.dim,
                                                                                    [self.bounds[1]] * self.dim)
        return acc
    
    def step(self):
        for index_positions in range(self.dim):
            candidates = [candidate for candidate in range(self.dim) if candidate != index_positions]
            a, b, c = self.best_positions[choice(candidates, 3, replace=False)]
            mutated = self.mutation([a, b, c])
            mutated = self.check_domain(mutated) #musi być zamiana 
            trial = self.crossover(mutated, index_positions)
            obj_target = self.func(self.best_positions[index_positions][np.newaxis])
            obj_trial = self.func(np.array(trial)[np.newaxis])
            if obj_trial < obj_target:
                # replace the target vector with the trial vector
                self.best_positions[index_positions] = trial
                # store the new objective function value
                self.best_scores[index_positions] = obj_trial
        self.update_global_best()
        self.record_scores()
   
    def optimize(self, iterations):
        # Start the timer
        start_time = time.time()
        runs = iterations
        for iteration in range(iterations):
            self.step()
            if math.isclose(self.func(np.array([self.global_best_position])), self.expected_min, abs_tol=1e-5):
                runs = iteration + 1
                break
        # Calculate the total execution time
        end_time = time.time()
        execution_time = end_time - start_time
        # x1 = np.linspace(0.0, iterations, num=iterations)
        # pyplot.plot(x1, self.best_scores_history, '.-')
        # pyplot.xlabel('Improvement Number')
        # pyplot.ylabel('Evaluation f(x)')
        # pyplot.show()

        return [self.global_best_position,  # 1
                self.func(np.array([self.global_best_position])),  # 2
                np.mean(self.best_scores_history),  # 3
                np.std(self.best_scores_history),  # 4
                np.mean(self.average_scores_history),  # 5
                np.mean(self.accuracy_history),  # 6
                np.std(self.accuracy_history),  # 7
                np.max(self.accuracy_history),  # 8
                runs,  # 9
                execution_time]  # 10

