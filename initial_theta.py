import numpy as np
from numpy import ndarray


def get_uniform_initial_theta(w: ndarray, omega: ndarray) -> ndarray:
    row_count, col_count = w.shape[0], w.shape[1]
    theta = np.zeros((row_count, col_count))

    non_zero_elements = [0] * row_count
    for i, row in enumerate(w):
        non_zero_elements[i] = sum(row)

    for i in range(row_count):
        for j in range(col_count):
            if w[i][j] == 1:
                theta[i][j] = 1 / non_zero_elements[i]  # равномерное распределение элементов матрицы

    return theta


def get_smart_initial_theta(w: ndarray, omega: ndarray) -> ndarray:
    row_count, col_count = w.shape[0], w.shape[1]
    theta = np.zeros((row_count, col_count))

    for i in range(row_count):
        for j in range(col_count):
            if w[i][j] == 1:
                theta[i][j] = omega[j]

    for i in range(row_count):
        s = sum(theta[i])
        for j in range(col_count):
            theta[i][j] /= s

    return theta
