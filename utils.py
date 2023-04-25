import random

import numpy as np


def get_random_w(n):
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                w[i][j] = random.randint(0, 1)
        w[i][random.randint(0, n - 1)] = 1

    return w
