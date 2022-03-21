import numpy as np
from ex2_utils import get_patch, generate_responses_1
import math


def ker_der(x_i, r):
    c = 1
    if np.abs(x) < r:
        return -2 * c * x_i
    return 0

def mean_shift_mode_seek(I, N, stop_criterion = 0.001):

    number_of_rows = len(I)
    number_of_columns = len(I[0])

    size_X = 5
    size_Y = 5

    ranX = range(-int(size_X/2 - 0.5), int(size_X/2 + 0.5))
    ranY = range(-int(size_Y/2 - 0.5), int(size_Y/2 + 0.5))

    Y = [[c for c in ranY] for r in ranX]
    X = [[r for c in ranY] for r in ranX]

    print(X)
    print(Y)


    x_k = 50.0
    y_k = number_of_rows/2
    
    for i in range(N):
        print(i)
        w_i = get_patch(I, [round(y_k), round(x_k)], [size_X, size_Y])[0]
        x_k_new = (np.sum(np.multiply(X, w_i)))/(np.sum(w_i))
        y_k_new = (np.sum(np.multiply(Y, w_i)))/(np.sum(w_i))
        
        x_k += x_k_new
        y_k += y_k_new
        
        if np.abs(x_k_new) < stop_criterion and np.abs(y_k_new) < stop_criterion:
            return round(x_k), round(y_k)

    return round(x_k), round(y_k)


res = generate_responses_1()
x_k = mean_shift_mode_seek(res, 1000)

print(x_k, res[x_k[0], x_k[1]])
