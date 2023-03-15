import numpy as np

from theta_formation import find_routing_matrix

if __name__ == '__main__':
    inp_omega = np.array([.35, .27, .15, .23])

    W = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0]
    ])

    Theta, out_omega = find_routing_matrix(inp_omega, W, log_step=100)
