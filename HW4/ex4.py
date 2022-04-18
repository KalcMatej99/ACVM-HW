import numpy as np
import math
from ex4_utils import kalman_step
import matplotlib.pyplot as plt
import sympy as sp


def traj1():
    N = 40
    v = np.linspace(5 * math.pi, 0, N)
    x = np.cos(v) * v
    y = np.sin(v) * v

    return x, y

def traj2():
    N = 40
    v = np.linspace(5 * math.pi, 0, N)
    x = np.sin(v) * v
    y = np.cos(v) * v

    return x, y


def traj3():
    N = 10 * 3
    x = []
    y = []
    for i in range(10):
        x.append((i + 1) * -1)
        y.append((i + 1) * -1)
        x.append(0)
        y.append((i + 1))
        x.append((i + 1) * 1)
        y.append((i + 1) * -1)

    return np.array(x), np.array(y)

def traj(trajGraph = traj1):

    fig, ax = plt.subplots(3, 5, figsize=(10, 10))

    for row, method in enumerate(["RW", "NCV", "NCA"]):
        col = 0
        for q, r in zip([100, 5, 1, 1, 1], [1, 1, 1, 5, 100]):

            x, y = trajGraph()
            sx = np . zeros((x . size, 1), dtype=np . float32) . flatten()
            sy = np . zeros((y . size, 1), dtype=np . float32) . flatten()
            sx[0] = x[0]
            sy[0] = y[0]

            T = 1.0
            if method == "RW":

                #T_ = sp.symbols('T')
                #F = sp.Matrix([[0, 0], [0, 0]])

                C = np.array([[1, 0],
                              [0, 1]])
                A = np.array([[1, 0],
                              [0, 1]])

                Q_i = np.array([[q, 0], [0, q]])
                R_i = np.array([[r, 0], [0, r]])

                state = np . zeros(
                    (A.shape[0], 1), dtype=np . float32) . flatten()
                state[0] = x[0]
                state[1] = y[0]

            elif method == "NCV":
                C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
                A = np.array([[1, T, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, T],
                              [0, 0, 0, 1]])

                #T, q = sp.symbols('T q')
                #Fi = sp.Matrix(A)
                #L = sp.Matrix([[0, 0],[1, 0],[0, 0],[0, 1]])
                #Q = sp.integrate( (Fi*L)*q*(Fi*L).T, (T, 0, T) )
                # print(Q)

                Q_i = np.array([[1.0*T*q, 1.0*T*q, 0, 0], [1.0*T*q, 1.0*T*q, 0, 0],
                               [0, 0, 1.0*T*q, 1.0*T*q], [0, 0, 1.0*T*q, 1.0*T*q]])
                R_i = np.array([[r, 0], [0, r]])

                state = np . zeros(
                    (A.shape[0], 1), dtype=np . float32) . flatten()
                state[0] = x[0]
                state[2] = y[0]
                state[1] = 0
                state[3] = 0

            else:

                F = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0], [
                    0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]])

                A = np.array([[1, T, T**2/2, 0, 0, 0], [0, 1, T, 0, 0, 0], [0, 0, 1, 0, 0, 0], [
                             0, 0, 0, 1, T, T**2/2], [0, 0, 0, 0, 1, T], [0, 0, 0, 0, 0, 1]])

                T_ = sp.symbols('T')
                F_ = sp.Matrix(F)
                Fi = sp.exp(F_*T_)
                #print("Fi", Fi)

                C = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
                
                T_, q_ = sp.symbols('T q')
                Fi_ = sp.Matrix(A)
                L_ = sp.Matrix([[0, 0],[0, 0],[1, 0],[0, 0],[0, 0],[0, 1]])
                Q = sp.integrate( (Fi_*L_)*q_*(Fi_*L_).T, (T_, 0, T_) )
                #print(Q)

                Q_i = np.array([[0.25*T*q, 0.5*T*q, 0.5*T*q, 0, 0, 0], [0.5*T*q, 1.0*T*q, 1.0*T*q, 0, 0, 0], [0.5*T*q, 1.0*T*q, 1.0*T*q, 0, 0, 0], [
                               0, 0, 0, 0.25*T*q, 0.5*T*q, 0.5*T*q], [0, 0, 0, 0.5*T*q, 1.0*T*q, 1.0*T*q], [0, 0, 0, 0.5*T*q, 1.0*T*q, 1.0*T*q]])
                R_i = np.array([[r, 0], [0, r]])
                state = np . zeros(
                    (A.shape[0], 1), dtype=np . float32) . flatten()
                state[0] = x[0]
                state[2] = 0
                state[1] = 0
                state[3] = y[0]
                state[4] = 0
                state[5] = 0

            covariance = np . eye(A.shape[0], dtype=np . float32)
            for j in range(1, x.size):
                state, covariance, _, _ = kalman_step(A, C, Q_i, R_i, np.reshape(
                    np.array([x[j], y[j]]), (-1, 1)), np.reshape(state, (-1, 1)), covariance)
                sx[j] = state[0]
                if method == "RW":
                    sy[j] = state[1]
                elif method == "NCV":
                    sy[j] = state[2]
                else:
                    sy[j] = state[3]

            ax[row, col].plot(x, y, "r", marker="o", label="Measurments")
            ax[row, col].plot(sx, sy, "b", marker="o", label="Method")
            ax[row, col].set_title(f"{method}: q = {q}, r = {r}")
            col += 1
    plt.show()


if __name__ == "__main__":
    traj(trajGraph = traj3)
