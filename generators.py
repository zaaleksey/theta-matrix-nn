import random

import numpy as np
from numpy import ndarray

from conjugate_gradient import conjugate
from gradient_descent import gradient_descent
from initial_theta import get_uniform_initial_theta


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


def generate_good_omegas(min_count, max_count, count, max_iter=2_000):
    good_omegas = []
    omegas = get_omegas(min_count, max_count, count)
    for omega in omegas:
        for _ in range(20):
            w = get_random_w(len(omega))
            _, _, _, it1 = gradient_descent(omega, w, get_uniform_initial_theta, max_it=max_iter)
            _, _, _, it2 = conjugate(omega, w, get_uniform_initial_theta, max_it=max_iter)

            if it1 != max_iter and it2 != max_iter and sum(omega) == 1:
                good_omegas.append(omega)
                break

    return good_omegas
