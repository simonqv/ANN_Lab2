import numpy as np
from matplotlib import pyplot as plt

import plotter

SIGMA2 = 1.8
EPOCH = 20



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

def make_phi_matrix(node_matrix, input_list):

    phi_matrix = np.zeros((len(input_list), len(node_matrix)))

    for i, x in enumerate(input_list):
        for j, node in enumerate(node_matrix):
            phi_matrix[i, j] = phi_i(x, node)

    return phi_matrix


def batch_least_squares(phi_mat, init_weights, train):
    #for node in range(len(phi_mat[0])):
    # calculate weight matrix with least square method
    phiT_phi = np.dot(phi_mat.T, phi_mat)

    phi_f = np.dot(phi_mat.T, train).reshape(-1, 1)
    print
    w = np.linalg.solve(phiT_phi, phi_f)
   # w_test= np.linalg.solve(phiT_phi, phi_f_test)
    #w, _ ,_ , _ = np.linalg.lstsq(phiT_phi, phi_f, rcond=None)
   # print(w_test.shape)
    #output = np.dot(phi_mat, w)
    #print(output-train)
    
    return w



def task1():
    col, train_sin, train_box, test_sin, test_box = generate_set()
    plotter.sin_and_box(col, train_sin, train_box, test_sin, test_box)
    # plt.show()

    # matrix small, medium, large
    m1, m2, m3 = make_node_matrix()
    plotter.points(m1, m2, m3)
    
    # train the network by adjusting weights (least square error)
    # Init weights
    init_w_m1 = np.random.normal(0, 0.5, len(m1))
    init_w_m2 = np.random.normal(0, 0.5, len(m2))
    init_w_m3 = np.random.normal(0, 0.5, len(m3))

    # phi matrix for sin
    phi_4_sin = make_phi_matrix(m1, train_sin)
    phi_12_sin = make_phi_matrix(m2, train_sin)
    phi_20_sin = make_phi_matrix(m3, train_sin)

    # phi matrix for box function
    phi_4_box = make_phi_matrix(m1, train_box)
    phi_12_box = make_phi_matrix(m2, train_box)
    phi_20_box = make_phi_matrix(m3, train_box)

    w_4_sin = batch_least_squares(phi_4_sin, init_w_m1, train_sin)
    w_12_sin = batch_least_squares(phi_12_sin, init_w_m2, train_sin)
    w_20_sin = batch_least_squares(phi_20_sin, init_w_m3, train_sin)

    w_4_box = batch_least_squares(phi_4_box, init_w_m1, train_box)
    w_12_box = batch_least_squares(phi_12_box, init_w_m2, train_box)
    w_20_box = batch_least_squares(phi_20_box, init_w_m3, train_box)

    # test on hold out set

    phi_4_sin_test  = make_phi_matrix(m1, test_sin)
    phi_12_sin_test  = make_phi_matrix(m2, test_sin)

    out_4 = np.dot(phi_4_sin_test, w_4_sin)

    out = np.dot(phi_12_sin, w_12_sin)
    out_test = np.dot(phi_12_sin_test, w_12_sin)
  
    x_axis = np.arange(0, 2*np.pi, 0.1)
    # plt.plot(x_axis, out)

    plt.plot(x_axis, out_4, label="4")

    plt.plot(np.arange(0, 2*np.pi, 0.1), out_test)
    plt.legend()
    plt.show()




def sin2x(x):
    return np.sin(2*x)


def box2x(x):
    return 1 if np.sin(2*x) >= 0 else -1


def phi_i(x, mu):
    phi = np.exp((- (np.linalg.norm(x - mu[:2]) ** 2) / (2*mu[2])))
    return phi


    # TODO: Make matrix of phi_i thiny.

   # plt.show()


def main(task):
    # Task 1: Batch mode training using least squares - supervised learning of network weights
    task1()

    # Task 2: Regression with noise


    # Task 3: Competitive learning (CL) to initialise RBF units
    return None


main(1)
