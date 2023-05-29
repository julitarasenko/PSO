import numpy as np
import math
import time
import random


class PSO:
    def __init__(self, n_particles, dim, bounds, test_func, expected_min, expected_x, i, qmc_interval, swarm_random=None, 
                 p_random=None, g_random=None, d_type=np.float64):
        self.n_particles = n_particles
        self.dim = dim
        self.bounds = bounds
        self.func = test_func
        self.expected_min = expected_min
        self.expected_x = expected_x
        self.swarm_random = swarm_random
        self.p_random = p_random
        self.g_random = g_random
        self.d_type = d_type
        self.positions = self.initialize_positions(i, qmc_interval)
        self.velocities = self.initialize_velocities()
        self.best_positions = self.positions.copy()
        self.best_scores = self.func(self.positions)
        self.global_best_position = None
        self.update_global_best()
        self.best_scores_history = []
        self.average_scores_history = []
        self.accuracy_history = []


    def initialize_positions(self, i, qmc_interval):
        if self.swarm_random is None:
            x_ = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_particles, self.dim)).astype(self.d_type)
        elif (i < qmc_interval[0] or i > qmc_interval[1]):
            x_ = self.swarm_random.rvs(size=(self.n_particles, self.dim)).astype(self.d_type)
        else:
            x_ = self.swarm_random(d=self.dim).random(n=self.n_particles).astype(self.d_type)
        # Transformation to a given domain
        old_min, old_max = x_.min(), x_.max()
        return ((x_ - old_min) / (old_max - old_min)) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

    def initialize_velocities(self):
        return np.random.standard_normal((self.n_particles, self.dim)).astype(self.d_type) #normal
        # np.np.random.uniform(-1, 1, (n_particles, dim)).astype(self.d_type) # uniform

    def generate_p(self, size):
        random.seed()
        if self.p_random is None:
            return np.random.rand(size, self.dim)
        else:
            return self.p_random.rvs(size=(size, self.dim)).astype(self.d_type)

    def generate_g(self, size):
        random.seed()
        if self.g_random is None:
            return np.random.rand(size, self.dim)
        else:
            return self.g_random.rvs(size=(size, self.dim)).astype(self.d_type)

    def update_velocities(self, w=0.8, c1=1, c2=1):
        p_rand = self.generate_p(self.n_particles)
        g_rand = self.generate_g(self.n_particles)
        # print(g_rand)
        self.velocities = (w * self.velocities) + (c1 * p_rand * (self.best_positions - self.positions)) + (
                c2 * g_rand * (self.global_best_position - self.positions))

    def update_positions(self):
        self.positions += self.velocities

    def update_best_positions(self):
        within_bounds = np.all((self.positions >= self.bounds[0]) & (self.positions <= self.bounds[1]), axis=1)
        scores = self.func(self.positions)
        # print(scores, "\n", "-----------------------","\n")
        better_scores_idx = (scores < self.best_scores) & within_bounds
        self.best_positions[better_scores_idx] = self.positions[better_scores_idx]
        self.best_scores[better_scores_idx] = scores[better_scores_idx]

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
        self.update_velocities()
        self.update_positions()
        self.update_best_positions()
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
