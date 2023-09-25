import numpy as np
from matplotlib import pyplot as plt

import plotter

SIGMA2 = 0.6 # necessary for 4 nodes but even better with higher
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


def batch_least_squares(phi_mat, train):
    # calculate weight matrix with least square method
    phiT_phi = np.dot(phi_mat.T, phi_mat)

    phi_f = np.dot(phi_mat.T, train).reshape(-1, 1)
    print
    w = np.linalg.solve(phiT_phi, phi_f)
    
    return w

def residual_err(output, targets):
    # average absolute difference between the network outputs and the desirable target values
    diff = np.abs(targets-output)
    avg_err = np.sum(diff)/len(output)
  
    return avg_err

def task1():
    col, train_sin, train_box, test_sin, test_box = generate_set()
    # matrix small, medium, large
    m1, m2, m3 = make_node_matrix()

    x = np.arange(0, 2*np.pi, 0.1)

    # plot true functions and starting points
    # plotter.sin_and_box(col, train_sin, train_box, test_sin, test_box)
    # plotter.points(m1, m2, m3, True, True, True)
    
    # train the network by adjusting weights (least square error)
    # Init weights

    # not necessary rn
    # init_w_m1 = np.random.normal(0, 0.5, len(m1))
    # init_w_m2 = np.random.normal(0, 0.5, len(m2))
    # init_w_m3 = np.random.normal(0, 0.5, len(m3))

    '''
    # ------ SIN --------

    # phi matrix for sin
    phi_4_sin = make_phi_matrix(m1, train_sin)
    phi_12_sin = make_phi_matrix(m2, train_sin)
    phi_20_sin = make_phi_matrix(m3, train_sin)

    # weight vectors
    w_4_sin = batch_least_squares(phi_4_sin, train_sin)
    w_12_sin = batch_least_squares(phi_12_sin, train_sin)
    w_20_sin = batch_least_squares(phi_20_sin, train_sin)
    
    # test on hold out set and sum to get output
    phi_4_sin_test  = make_phi_matrix(m1, test_sin)
    phi_12_sin_test  = make_phi_matrix(m2, test_sin)
    phi_20_sin_test  = make_phi_matrix(m3, test_sin)

    out_4_sin = np.dot(phi_4_sin_test, w_4_sin)
    out_12_sin = np.dot(phi_12_sin_test, w_12_sin)
    out_20_sin = np.dot(phi_20_sin_test, w_20_sin)

    # plot the results
    x = np.arange(0, 2*np.pi, 0.1)
    # true line
    plotter.plot_line(col, test_sin, "True line")
    plotter.plot_line(x, out_4_sin, "4 nodes")
    #plotter.plot_line(x, out_12_sin, "12 nodes")
    #plotter.plot_line(x, out_20_sin, "20 nodes")
    plotter.points(m1, m2, m3, True)
    plt.legend()
    plt.show()

    # residual error 
    err_4_sin = residual_err(out_4_sin, test_sin.reshape(-1,1))
    err_12_sin = residual_err(out_12_sin, test_sin.reshape(-1, 1))
    err_20_sin = residual_err(out_20_sin, test_sin.reshape(-1, 1))

    print(f"--- Absolute residual error (sin) ---\n 4 nodes: {err_4_sin} \n 12 nodes: {err_12_sin} \n 20 nodes: {err_20_sin}\n")

    '''
    
    # ------ BOX --------

    # TODO: Figure out why box is either not working or perfect when changing sigma
    
    m0 = np.array([[3.8, 0.0, SIGMA2], [0.64, 0.58, SIGMA2]])
    phi_2_box = make_phi_matrix(m0, train_box)
    w_2_box = batch_least_squares(phi_2_box, train_box)
    phi_2_box_test = make_phi_matrix(m0, test_box)
    out_2_box = np.dot(phi_2_box_test, w_2_box)
    plotter.plot_line(x, out_2_box, "2 nodes")
    plt.scatter(m0[:,0], m0[:, 1])
    err_2_box = residual_err(out_2_box, test_box.reshape(-1,1))
    print(err_2_box)


    # phi matrix for box function
    phi_4_box = make_phi_matrix(m1, train_box)
    phi_12_box = make_phi_matrix(m2, train_box)
    phi_20_box = make_phi_matrix(m3, train_box)

    # calculate weights from rbf to output
    w_4_box = batch_least_squares(phi_4_box, train_box)
    w_12_box = batch_least_squares(phi_12_box, train_box)
    w_20_box = batch_least_squares(phi_20_box, train_box)

    # test on hold out
    phi_4_box_test = make_phi_matrix(m1, test_box)
    phi_12_box_test = make_phi_matrix(m2, test_box)
    phi_20_box_test = make_phi_matrix(m3, test_box)

    out_4_box = np.dot(phi_4_box_test, w_4_box)
    out_12_box = np.dot(phi_12_box_test, w_12_box)
    out_20_box = np.dot(phi_20_box_test, w_20_box)

    # plot result
    #plotter.plot_line(x, out_4_box, "4 nodes")
    #plotter.plot_line(x, out_12_box, "12 nodes")
    #plotter.plot_line(x, out_20_box, "20 nodes")
    plotter.plot_line(col, train_box, "True line")
    plotter.points(m1, m2, m3, True)
    plt.legend()
    plt.show()

    # residual error 
    err_4_box = residual_err(out_4_box, test_box.reshape(-1,1))
    err_12_box = residual_err(out_12_box, test_box.reshape(-1, 1))
    err_20_box = residual_err(out_20_box, test_box.reshape(-1, 1))

    print(f"--- Absolute residual error (box) ---\n 4 nodes: {err_4_box} \n 12 nodes: {err_12_box} \n 20 nodes: {err_20_box}\n")




# ------------- HELPERS ---------------

def sin2x(x):
    return np.sin(2*x)


def box2x(x):
    return 1 if np.sin(2*x) >= 0 else -1


def phi_i(x, mu):
    phi = np.exp((- (np.linalg.norm(x - mu[:2]) ** 2) / (2*mu[2])))
    return phi


def main(task):
    match task:
        case 1:
            # Task 1: Batch mode training using least squares - supervised learning of network weights
            print("----------------\n--- Task 3.1 ---\n---------------- ")
            task1()
        case 2:
            print("----------------\n--- Task 3.2 ---\n---------------- ")
            return 0
        case 3:
            print("----------------\n--- Task 3.3 ---\n---------------- ")
            return 0

    # Task 2: Regression with noise


    # Task 3: Competitive learning (CL) to initialise RBF units
    return None


main(1)
