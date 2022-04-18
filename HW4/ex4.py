import numpy as np
import math
from ex4_utils import kalman_step
import matplotlib.pyplot as plt
import sympy


def traj(method = "RW", q = 1.0, r = 1.0):

    N = 40
    v = np.linspace(5 * math.pi, 0, N)
    x = np.cos(v) * v
    y = np.sin(v) * v

    T = 1.0
    if method == "RW":
        A = np.array([[1.0, T], [1.0, 0.0]])
        Q_i = np.array([[q/3, q/2], [q/2, q]])
        C = np.array([[1, 0], [0, 1]])
        R_i = np.array([[r, 0], [0, r]])
    elif method == "NCV":
        A = np.array([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
        C = np.array([1,0,0,0],[0,0,1,0])
        F = np.array([[1, 0, T, 0],
            [0, 1, 0, T],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        L = np.array([[0,0], [1,0], [0,0],[0,1]])


    sx = np . zeros((x . size, 1), dtype=np . float32) . flatten()
    sy = np . zeros((y . size, 1), dtype=np . float32) . flatten()
    sx[0] = x[0]
    sy[0] = y[0]
    state = np . zeros((A.shape[0], 1), dtype=np . float32) . flatten()
    state[0] = x[0]
    state[2] = y[0]
    state[1] = 0
    state[3] = 0
    covariance = np . eye(A.shape[0], dtype=np . float32)
    for j in range(1, x.size):
        state, covariance, _, _ = kalman_step(A, C, Q_i, R_i, np.reshape(
            np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)
        sx[j] = state[0]
        sy[j] = state[1]

    plt.plot(x, y)
    plt.plot(sx, sy)
    plt.title(f"RW: q = {q}, r = {r}")
    plt.show()
