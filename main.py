import numpy as np

from theta_formation import find_routing_matrix, get_uniform_initial_theta

if __name__ == '__main__':
    inp_omega = np.array([.35, .27, .15, .23])

    w = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0]
    ])

    # приближенное значение
    # w = np.array([
    #     [0, 0, 1, 1],
    #     [1, 0, 1, 1],
    #     [0, 0, 0, 1],
    #     [1, 1, 0, 0]
    # ])

    Theta, out_omega = find_routing_matrix(inp_omega, w, get_uniform_initial_theta)
