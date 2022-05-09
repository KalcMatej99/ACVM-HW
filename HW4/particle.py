import numpy as np
from other.ex2_utils import get_patch, Tracker, create_epanechnik_kernel, extract_histogram
import sympy as sp


class ParticleFilter(Tracker):

    def __init__(self, params):
        self.sigma = 4
        self.sigma_weights = 1
        self.alpha = 0.01
        self.n_bins = 16
        self.number_of_particles = 100
        self.T = 1
        self.q = 10

        self.dynamic_model = params["dynamic_model"]

        print(self.dynamic_model)

    def initialize(self, image, region):

        region = np.array(region).astype(int)

        if region[2] % 2 == 0:
            region[2] += 1
        if region[3] % 2 == 0:
            region[3] += 1
        self.position = [region[0] + region[2] / 2, region[1] + region[3] / 2]
        self.size = [region[2], region[3]]

        if self.dynamic_model == "RW":

            T_, q_ = sp.symbols('T q')
            F_ = sp.Matrix([[0, 0], [0, 0]])
            Fi_ = sp.exp(F_*T_)

            L_ = sp.Matrix([[1, 0], [0, 1]])
            Q_ = sp.integrate((Fi_*L_)*q_*(Fi_*L_).T, (T_, 0, T_))

            Fi_ = Fi_.subs(T_, self.T)
            Q_ = Q_.subs(T_, self.T)
            Q_ = Q_.subs(q_, self.q)

            self.A = np.array(Fi_, dtype=np.float32)
            self.Q_i = np.array(Q_, dtype=np.float32)
            self.particles = np.random.multivariate_normal(
                [self.position[0], self.position[1]], self.Q_i, self.number_of_particles)
        elif self.dynamic_model == "NCV":

            T_, q_ = sp.symbols('T q')
            F_ = sp.Matrix([[0, 1, 0, 0], [0, 0, 0, 0],
                           [0, 0, 0, 1], [0, 0, 0, 0]])
            Fi_ = sp.exp(F_*T_)
            L = sp.Matrix([[0, 0], [1, 0], [0, 0], [0, 1]])
            Q_ = sp.integrate((Fi_*L)*q_*(Fi_*L).T, (T_, 0, T_))

            Fi_ = Fi_.subs(T_, self.T)
            Q_ = Q_.subs(T_, self.T)
            Q_ = Q_.subs(q_, self.q)

            self.A = np.array(Fi_, dtype=np.float32)
            self.Q_i = np.array(Q_, dtype=np.float32)

            self.particles = np.random.multivariate_normal(
                [self.position[0], 0, self.position[1], 0], self.Q_i, self.number_of_particles)

        elif self.dynamic_model == "NCA":

            F = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0], [
                0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]])

            T_, q_ = sp.symbols('T q')
            F_ = sp.Matrix(F)
            Fi_ = sp.exp(F_*T_)

            L = sp.Matrix(
                [[0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 1]])
            Q_ = sp.integrate((Fi_*L)*q_*(Fi_*L).T, (T_, 0, T_))

            Fi_ = Fi_.subs(T_, self.T)
            Q_ = Q_.subs(T_, self.T)
            Q_ = Q_.subs(q_, self.q)

            self.A = np.array(Fi_, dtype=np.float32)
            self.Q_i = np.array(Q_, dtype=np.float32)

            self.particles = np.random.multivariate_normal(
                [self.position[0], 0, 0, self.position[1], 0, 0], self.Q_i, self.number_of_particles)

        self.weights = np.ones(self.number_of_particles)

        self.kernel = create_epanechnik_kernel(
            self.size[1], self.size[0], self.sigma)

        patch, _ = get_patch(image, self.position, self.size)
        self.h_tar = extract_histogram(patch, self.n_bins, self.kernel)

        return self.particles[:, self.indexOfXYinParticle()]

    def H(self, p, q):
        return np.sqrt(np.sum(np.square(np.sqrt(p) - np.sqrt(q)))/2)

    def indexOfXYinParticle(self):
        if self.dynamic_model == "RW":
            return [0, 1]
        elif self.dynamic_model == "NCV":
            return [0, 2]
        elif self.dynamic_model == "NCA":
            return [0, 3]
        return None

    def track(self, image):

        weights_norm = self.weights / np.sum(self.weights)
        weights_cumsumed = np.cumsum(weights_norm)
        rand_samples = np.random.rand(self.number_of_particles, 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed)
        self.particles = self.particles[sampled_idxs.flatten(), :]

        self.particles = np.matmul(self.A, self.particles.T).T + np.random.multivariate_normal(
            np.zeros(len(self.particles[0])), self.Q_i, self.number_of_particles)

        for i, particle in enumerate(self.particles):
            particle_position = [particle[0],
                                 particle[self.indexOfXYinParticle()[1]]]
            if particle_position[0] > self.position[0] - self.size[0]/2 and particle_position[0] < self.position[0] + self.size[0]/2  and particle_position[1] > self.position[1] - self.size[1]/2  and particle_position[1] < self.position[1] + self.size[1]/2 :
                patch, _ = get_patch(image, particle_position, self.size)
                if self.kernel.shape != patch.shape[:2]:
                    self.kernel = create_epanechnik_kernel(patch.shape[1], patch.shape[0], self.sigma)[
                        :patch.shape[0], :patch.shape[1]]

                h_i = extract_histogram(patch, self.n_bins, self.kernel)
                self.weights[i] = np.exp(
                    (self.H(h_i, self.h_tar)**2)/(-2 * self.sigma_weights ** 2))
            else:
                self.weights[i] = 0

        if np.sum(self.weights) > 0:
            self.weights = self.weights / np.sum(self.weights)

        xyPosOfParticles = np.array(
            [self.particles[:, 0], self.particles[:, self.indexOfXYinParticle()[1]]]).T
        self.position = np.dot(self.weights, xyPosOfParticles)

        patch, _ = get_patch(image, self.position, self.size)
        if self.kernel.shape != patch.shape[:2]:
            self.kernel = create_epanechnik_kernel(patch.shape[1], patch.shape[0], self.sigma)[
                :patch.shape[0], :patch.shape[1]]
        self.h_tar = (1 - self.alpha) * self.h_tar + self.alpha * \
            extract_histogram(patch, self.n_bins, self.kernel)

        return [self.position[0] - self.size[0]/2, self.position[1] - self.size[1]/2, self.size[0], self.size[1]], self.particles[:, self.indexOfXYinParticle()]
