import random

import numpy as np
from numpy import ndarray


def get_random_w(n: int) -> ndarray:
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                w[i][j] = random.randint(0, 1)
        w[i][random.randint(0, n - 1)] = 1

    return w


def get_omegas(min_systems: int, max_systems: int, count: int) -> list[ndarray]:
    omegas = []
    for systems in range(min_systems, max_systems + 1):
        for _ in range(count):
            temp = [random.randint(5, 10) for _ in range(systems)]
            omega = [round(item / sum(temp), 2) for item in temp]
            diff = 1 - sum(omega)
            omega[random.randrange(len(omega))] += diff
            omegas.append(np.array(omega))

    return omegas
