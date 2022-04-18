import numpy as np
from other.ex2_utils import get_patch, Tracker, create_epanechnik_kernel, extract_histogram
import cv2
import random

class ParticleFilter(Tracker):

    def __init__(self):
        self.sigma = 1
        self.alpha = 0.01
        self.n_bins = 16
        self.number_of_particles = 100
        self.sigma_gaussian_particles = 1
        self.T = 1
        self.q = 100

    def initialize(self, image, region):

        if region[2] % 2 == 0:
            region[2] += 1
        if region[3] % 2 == 0:
            region[3] += 1
        self.position = [region[0] + region[2] / 2, region[1] + region[3] / 2]
        self.size = [region[2], region[3]]

        self.Q_i = np.array([[self.T*self.q, self.T*self.q, 0, 0], [self.T*self.q, self.T*self.q, 0, 0],
                        [0, 0, self.T*self.q, self.T*self.q], [0, 0, self.T*self.q, self.T*self.q]])
        self.particles = np.random.multivariate_normal([self.position[0], 0, self.position[1], 0], self.Q_i, self.number_of_particles)
        self.weights = np.ones(self.number_of_particles)

        self.A = np.array([[1, self.T, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, self.T],
                        [0, 0, 0, 1]])

        self.kernel = create_epanechnik_kernel(self.size[1], self.size[0], self.sigma)

        patch, _ = get_patch(image, self.position, self.size)
        self.h_tar = extract_histogram(patch, self.n_bins, self.kernel)

    def H(self, p, q):
        return (1.0 / np.sqrt(2.0)) * np.sqrt(np.sum( np.square(np.sqrt(p) - np.sqrt(q))))

    def track(self, image):

        weights_norm = self.weights / np.sum(self.weights)
        weights_cumsumed = np.cumsum(weights_norm)
        rand_samples = np.random.rand(self.number_of_particles, 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed)
        self.particles = self.particles[sampled_idxs.flatten(), : ]

        particles_moved = np.matmul(self.A, self.particles.T).T
        particles_moved = np.array([np.random.multivariate_normal(particle_moved, self.Q_i, 1)[0] for particle_moved in particles_moved])

        sigma = 2


        for i, particle in enumerate(particles_moved):
            patch, _ = get_patch(image, [particle[0], particle[2]], self.size)
            h_i = extract_histogram(patch, self.n_bins, self.kernel)
            self.weights[i] = np.exp((self.H(h_i, self.h_tar)**2)/(-2 * sigma ** 2))

        print(self.weights)
        self.weights = self.weights / np.sum(self.weights)

        xyPosOfParticles = np.array([particles_moved[:, 0], particles_moved[:, 2]]).T
        self.position = np.dot(self.weights, xyPosOfParticles)


        print(self.position)
        patch, _ = get_patch(image, self.position, self.size)
        self.h_tar = (1 - self.alpha) * self.h_tar + self.alpha * extract_histogram(patch, self.n_bins, self.kernel)


        return [self.position[0] - self.size[0]/2, self.position[1] - self.size[1]/2, self.size[0], self.size[1]]

