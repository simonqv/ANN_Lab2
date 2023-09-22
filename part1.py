import numpy as np
from matplotlib import pyplot as plt

import plotter

SIGMA2 = 0.1


def sin2x(x):
    return np.sin(2*x)


def box2x(x):
    return 1 if np.sin(2*x) >= 0 else -1


def phi_i(x, mu):
    phi = np.exp((- (np.linalg.norm(x - mu[:2]) ** 2) / (2*mu[2])))
    return phi


def generate_set():
    col = np.arange(0, 2 * np.pi, 0.1).reshape((-1, 1))
    train_set_sin = []
    train_set_box = []
    test_set_sin = []
    test_set_box = []
    for x in col:
        train_set_sin.append(sin2x(x))
        train_set_box.append(box2x(x))

        test_set_sin.append(sin2x(x + 0.05))
        test_set_box.append(box2x(x + 0.05))

    return col, np.array(train_set_sin), np.array(train_set_box), np.array(test_set_sin), np.array(test_set_box)


def make_node_matrix():
    """
    [[x, y, sigma]] where is x, y is coordinate for mean
    :return: nodes
    """
    four = np.array([[0.76, 0.98, SIGMA2], [2.33, -0.98, SIGMA2], [3.9, 0.98, SIGMA2], [5.5, -0.98, SIGMA2]])
    twelve = np.array([[0.12, 0.3, SIGMA2], [1.13, 0.96, SIGMA2], [1.46, 0.3, SIGMA2], [1.6, -0.3, SIGMA2], [2.0, -0.9, SIGMA2], [2.7, -0.9, SIGMA2], [3.1, 0.0, SIGMA2], [3.3, 0.6, SIGMA2], [3.9, 0.98, SIGMA2], [4.6, 0.5, SIGMA2], [4.8, -0.34, SIGMA2], [5.9, -0.9, SIGMA2]])
    twenty = np.array([[0.1, 0.2, SIGMA2],
                       [0.01, 0.98, SIGMA2],
                       [1.13, 0.96, SIGMA2],
                       [1.46, 0.3, SIGMA2],
                       [1.5, 0.98, SIGMA2],
                       [1.6, -0.3, SIGMA2],
                       [1.61, -0.98, SIGMA2],
                       [2.0, -0.9, SIGMA2],
                       [2.7, -0.9, SIGMA2],
                       [3, -0.98, SIGMA2],
                       [3.1, 0.0, SIGMA2],
                       [3.2, 0.98, SIGMA2],
                       [3.3, 0.6, SIGMA2],
                       [3.9, 0.98, SIGMA2],
                       [4.6, 0.5, SIGMA2],
                       [4.61, 0.97, SIGMA2],
                       [4.8, -0.34, SIGMA2], [4.79, -0.98, SIGMA2], [5.9, -0.9, SIGMA2], [6.2, -0.98, SIGMA2]])

    return four, twelve, twenty


def task1():
    col, train_sin, train_box, test_sin, test_box = generate_set()
    plotter.sin_and_box(col, train_sin, train_box, test_sin, test_box)
    # plt.show()

    # matrix small, medium, large
    m1, m2, m3 = make_node_matrix()
    plotter.points(m1, m2, m3)

    # TODO: Make matrix of phi_i thiny.

    plt.show()


def main(task):
    # Task 1: Batch mode training using least squares - supervised learning of network weights
    task1()

    # Task 2: Regression with noise


    # Task 3: Competitive learning (CL) to initialise RBF units
    return None


main(1)
