from pprint import pprint

import numpy as np

from theta_formation import find_routing_matrix, get_uniform_initial_theta, get_initial_theta
from utils import get_random_w

if __name__ == '__main__':
    max_it = 10_000
    max_step = 100
    inp_omega = np.array([.35, .27, .15, .23])

    # w = np.array([
    #     [0, 1, 1, 1],
    #     [1, 0, 1, 0],
    #     [0, 0, 0, 1],
    #     [1, 1, 0, 0]
    # ])

    # приближенное значение
    # w = np.array([
    #     [0, 0, 1, 1],
    #     [1, 0, 1, 1],
    #     [0, 0, 0, 1],
    #     [1, 1, 0, 0]
    # ])
    difference = []
    for step in range(max_step):
        w = get_random_w(len(inp_omega))
        # theta, out_omega, _ = find_routing_matrix(inp_omega, w, get_uniform_initial_theta)
        # print("Начальная омега:", inp_omega)
        # print("Полученная омега:", out_omega, sum(out_omega))
        # print("Theta\n", theta, "\n")
        #
        # theta, out_omega, _ = find_routing_matrix(inp_omega, w, get_initial_theta)
        # print("Начальная омега:", inp_omega)
        # print("Полученная омега:", out_omega, sum(out_omega))
        # print("Theta\n", theta, "\n")
        # print("#" * 150)
        _, _, it = find_routing_matrix(inp_omega, w, get_uniform_initial_theta, max_it=max_it)
        _, _, opt_it = find_routing_matrix(inp_omega, w, get_initial_theta, max_it=max_it)
        print()

        if it != max_it or opt_it != max_it:
            difference.append(it - opt_it)

    print(len(difference))
    pprint(difference)
