import numpy as np
from matplotlib import pyplot as plt
import plotter

SIGMA2 = 0.5  # necessary for 4 nodes but even better with higher
EPOCH = 200
ETA = 0.1


def generate_set(noise=False):
    col = np.arange(0, 2 * np.pi, 0.1).reshape((-1, 1))

    train_set_sin = np.empty((0, 1))
    train_set_box = np.empty((0, 1))
    test_set_sin = np.empty((0, 1))
    test_set_box = np.empty((0, 1))
    for x in col:
        train_set_sin = np.append(train_set_sin, sin2x(x))
        train_set_box = np.append(train_set_box, box2x(x))

        test_set_sin = np.append(test_set_sin, sin2x(x + 0.05))
        test_set_box = np.append(test_set_box, box2x(x + 0.05))
    if noise:
        noise_list = np.random.normal(0, 0.1, (63, 4))

        train_set_sin += noise_list[:, 0]
        train_set_box += noise_list[:, 1]

        test_set_sin += noise_list[:, 2]
        test_set_box += noise_list[:, 3]

    return col, train_set_sin, train_set_box, test_set_sin, test_set_box


def make_node_matrix():
    """
    [[x, y, sigma]] where is x, y is coordinate for mean
    :return: nodes
    """
    four = np.array([[0.76, 0.98, SIGMA2], [2.33, -0.98, SIGMA2], [3.9, 0.98, SIGMA2], [5.5, -0.98, SIGMA2]])
    eight = np.array(([[0.76, 0.98, SIGMA2], [2.33, -0.98, SIGMA2], [3.9, 0.98, SIGMA2], [5.5, -0.98, SIGMA2], [4.640, 0.049, SIGMA2], [3.093, -0.010, SIGMA2], [1.526, -0.019, SIGMA2], [0.130, 0.290, SIGMA2]]))
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
    
    return four, twelve, eight, twenty


def make_phi_matrix(node_matrix, input_list, col):
    phi_matrix = np.zeros((len(input_list), len(node_matrix)))
    for i, x in enumerate(input_list):
        for j, node in enumerate(node_matrix):
            phi_matrix[i, j] = phi_i(x, node, col[i])

    return phi_matrix


def batch_least_squares(phi_mat, train):
    # calculate weight matrix with least square method
    phiT_phi = np.dot(phi_mat.T, phi_mat)

    phi_f = np.dot(phi_mat.T, train).reshape(-1, 1)

    w, _, _, _ = np.linalg.lstsq(phiT_phi, phi_f)

    return w


def residual_err(output, targets):
    # average absolute difference between the network outputs and the desirable target values
    diff = np.abs(targets - output)
    avg_err = np.sum(diff) / len(output)

    return avg_err


def sequential_delta(input_x, label, rbf_nodes, weights, input_x_list):
    '''
    expected error e ~ instantaneous error ê = 
     = 0.5 * (f(latest pattern) - f^(latest pattern))^2 = 0.5error^2 
    '''
    phi_x = make_phi_matrix(rbf_nodes, input_x, input_x_list)  # input is scalar so phi_x is 1xnodes so transpose needed
    e = label - np.dot(phi_x, weights)
    delta_w = ETA * e * phi_x
    # delta_w becomes 1xnodes
    return delta_w.T


def weight_update(x_k, y_k, nodes_lists, w_m1, w_m2, w_m3, input_x):
    w_m1 = sequential_delta(x_k, y_k, nodes_lists[0], w_m1, input_x)
    w_m2 = sequential_delta(x_k, y_k, nodes_lists[1], w_m2, input_x)
    w_m3 = sequential_delta(x_k, y_k, nodes_lists[2], w_m3, input_x)
    return w_m1, w_m2, w_m3


def test_8_nodes():
    #np.random.seed(1)
    input_x, train_sin, _, test_sin, _ = generate_set(True)
    #x, train_sin_true, _, test_sin, _ = generate_set(True)

    #shuffle input points
    #input_x, train_sin, index_list = shuffle_data(x, train_sin_true)
    
    _, _, nodes, _ = make_node_matrix()
    # init weights
    weights = np.random.normal(0, 0.2, len(nodes)).reshape(-1, 1)

    # learning loop
    for epoch in range(EPOCH):
        for k, x_k in enumerate(input_x):
            delta_w = sequential_delta(x_k, train_sin[k], nodes, weights, input_x)
            weights += delta_w
        #input_x, train_sin, index_list = shuffle_data(input_x, train_sin)


    # 63x4 * 4x1 = 63x1
    preds = make_phi_matrix(nodes, test_sin, input_x+0.05)
    preds = np.dot(preds, weights)
    
    #preds = reverse_shuffle(preds, index_list)
   
    #plotter.plot_line(x, preds)
    #plotter.plot_line(x, train_sin_true)
    plotter.plot_line(input_x, preds)
    plotter.plot_line(input_x, train_sin)
    plt.show()
        


def task1():
   
    # TODO: How can you simply transform the output of your RBF
    # network to reduce the residual error to 0 for the 
    # square(2x) problem? 

    col, train_sin, train_box, test_sin, test_box = generate_set()
    # matrix small, medium, large
    m1, m2, m4, m3 = make_node_matrix()

    # make a graph with just all points
    plt.figure(1)
    plt.title("4 RBF units")
    plotter.points(m1, m2, m3, p1=True)
    plotter.plot_line(col, test_sin)
    plotter.plot_line(col, test_box)

    plt.figure(2)
    plt.title("8 RBF units")
    for x in m4:
        plt.scatter(x[0], x[1], c="r", marker="x")
    plotter.plot_line(col, test_sin)
    plotter.plot_line(col, test_box)

    plt.figure(3)
    plt.title("12 RBF units")
    plotter.points(m1, m2, m3, p2=True)
    plotter.plot_line(col, test_sin)
    plotter.plot_line(col, test_box)

    plt.figure(4)
    plt.title("20 RBF units")
    plotter.points(m1, m2, m3, p3=True)
    plotter.plot_line(col, test_sin)
    plotter.plot_line(col, test_box)

    plt.show()

    x = np.arange(0, 2 * np.pi, 0.1)

    # plot true functions and starting points
    plt.figure(1)
    plt.title("Training and Test Curves, and Selected RBF Nodes")
    plotter.sin_and_box(col, train_sin, train_box, test_sin, test_box)
    plotter.points(m1, m2, m3, True, True, True)
    plt.legend()
    plt.show()

    # train the network by adjusting weights (least square error)
    # Init weights

    # not necessary rn
    # init_w_m1 = np.random.normal(0, 0.5, len(m1))
    # init_w_m2 = np.random.normal(0, 0.5, len(m2))
    # init_w_m3 = np.random.normal(0, 0.5, len(m3))


    # ------ SIN --------

    # phi matrix for sin
    phi_4_sin = make_phi_matrix(m1, train_sin, col)
    phi_8_sin = make_phi_matrix(m4, train_sin, col)
    phi_12_sin = make_phi_matrix(m2, train_sin, col)
    phi_20_sin = make_phi_matrix(m3, train_sin, col)


    # weight vectors
    w_4_sin = batch_least_squares(phi_4_sin, train_sin)
    w_12_sin = batch_least_squares(phi_12_sin, train_sin)
    w_8_sin = batch_least_squares(phi_8_sin, train_sin)
    w_20_sin = batch_least_squares(phi_20_sin, train_sin)

    # test on hold out set and sum to get output

    phi_4_sin_test = make_phi_matrix(m1, test_sin, col + 0.05)
    phi_8_sin_test  = make_phi_matrix(m4, test_sin, col + 0.05)
    phi_12_sin_test = make_phi_matrix(m2, test_sin, col + 0.05)
    phi_20_sin_test = make_phi_matrix(m3, test_sin, col + 0.05)

    out_4_sin = np.dot(phi_4_sin_test, w_4_sin)
    out_12_sin = np.dot(phi_12_sin_test, w_12_sin)
    out_8_sin = np.dot(phi_8_sin_test, w_8_sin)
    out_20_sin = np.dot(phi_20_sin_test, w_20_sin)

    # plot the results
    x = np.arange(0, 2 * np.pi, 0.1)
    plt.figure("Sin prediction")
    # true line
    plotter.plot_line(col, test_sin, "True line", "-")

    # Predicted lines
    plotter.plot_line(x, out_4_sin, "4 nodes", ":")
    plotter.plot_line(x, out_12_sin, "12 nodes", "--")
    plotter.plot_line(x, out_20_sin, "20 nodes", "-.")
    plotter.points(m1, m2, m3, True, True, True)
    plt.legend()
    # plt.show()

    # residual error
    err_4_sin = residual_err(out_4_sin, test_sin.reshape(-1, 1))
    err_12_sin = residual_err(out_12_sin, test_sin.reshape(-1, 1))
    err_8_sin = residual_err(out_8_sin, test_sin.reshape(-1, 1))
    err_20_sin = residual_err(out_20_sin, test_sin.reshape(-1, 1))

    print(f"--- Absolute residual error (sin) ---\n 4 nodes: {err_4_sin} \n 8 nodes: {err_8_sin} \n 12 nodes: {err_12_sin} \n 20 nodes: {err_20_sin}\n")

    # ------ BOX --------

    plt.figure("Box prediction")

    # phi matrix for box function
    phi_4_box = make_phi_matrix(m1, train_box, col)
    phi_8_box = make_phi_matrix(m4, train_box, col)
    phi_12_box = make_phi_matrix(m2, train_box, col)
    phi_20_box = make_phi_matrix(m3, train_box, col)

    # calculate weights from rbf to output
    w_4_box = batch_least_squares(phi_4_box, train_box)
    w_8_box = batch_least_squares(phi_8_box, train_box)
    w_12_box = batch_least_squares(phi_12_box, train_box)
    w_20_box = batch_least_squares(phi_20_box, train_box)

    # test on hold out
    phi_4_box_test = make_phi_matrix(m1, test_box, col + 0.05)
    phi_8_box_test = make_phi_matrix(m4, test_box, col + 0.05)
    phi_12_box_test = make_phi_matrix(m2, test_box, col + 0.05)
    phi_20_box_test = make_phi_matrix(m3, test_box, col + 0.05)

    out_4_box = np.dot(phi_4_box_test, w_4_box)
    out_8_box = np.dot(phi_8_box_test, w_8_box)
    out_12_box = np.dot(phi_12_box_test, w_12_box)
    out_20_box = np.dot(phi_20_box_test, w_20_box)

    # true line
    plotter.plot_line(col, train_box, "True line", "-")

    # plot result (predictions)
    plotter.plot_line(x + 0.05, out_4_box, "4 nodes", ":")
    plotter.plot_line(x + 0.05, out_12_box, "12 nodes", "--")
    plotter.plot_line(x + 0.05, out_20_box, "20 nodes", "-.")
    plotter.points(m1, m2, m3, True, True, True)
    plt.legend()
    plt.show()

    # residual error 
    err_4_box = residual_err(out_4_box, test_box.reshape(-1, 1))
    err_8_box = residual_err(out_8_box, test_box.reshape(-1,1))
    err_12_box = residual_err(out_12_box, test_box.reshape(-1, 1))
    err_20_box = residual_err(out_20_box, test_box.reshape(-1, 1))

    print(f"--- Absolute residual error (box) ---\n 4 nodes: {err_4_box} \n 8 nodes: {err_8_box} \n 12 nodes: {err_12_box} \n 20 nodes: {err_20_box}\n")


def task2():
    # generate a noisy dataset, noise for both train and test
    input_x, train_sin, train_box, test_sin, test_box = generate_set(noise=True)

    #shuffle input points
    input_x, train_sin, index_list = shuffle_data(input_x, train_sin)
    
    #reorder them back again
    ordered_y = reverse_shuffle(input_x, train_sin, index_list)
   

    x = np.arange(0, 2 * np.pi, 0.1)
    plotter.plot_line(x, ordered_y, "True noisy line sin")
    #plotter.plot_line(input_x, train_box, "True noisy line box")
    plt.legend()
    #plt.show()

    nodes_lists = list(make_node_matrix())

    plotter.points(nodes_lists[0], nodes_lists[2], nodes_lists[1], True, False, True)

    # TODO: Make copies for BOX later
    w_m1 = np.random.normal(0, 0.2, len(nodes_lists[0])).reshape(-1, 1)
    w_m2 = np.random.normal(0, 0.2, len(nodes_lists[1])).reshape(-1, 1)
    w_m3 = np.random.normal(0, 0.2, len(nodes_lists[2])).reshape(-1, 1)

    # learning loop
    for epoch in range(EPOCH):
        for k, x_k in enumerate(input_x):
            d_w_m1, d_w_m2, d_w_m3 = weight_update(x_k, train_sin[k], nodes_lists, w_m1, w_m2, w_m3, input_x)
            w_m1 += d_w_m1
            w_m2 += d_w_m2
            w_m3 += d_w_m3

    # 63x4 * 4x1 = 63x1
    pred_1_sin = np.dot(make_phi_matrix(nodes_lists[0], test_sin, input_x+0.05), w_m1)
    pred_2_sin = np.dot(make_phi_matrix(nodes_lists[1], test_sin, input_x+0.05), w_m2)
    pred_3_sin = np.dot(make_phi_matrix(nodes_lists[2], test_sin, input_x+0.05), w_m3)

    #shuffle predictions to original order
    pred_1_sin = reverse_shuffle(input_x, pred_1_sin, index_list)
    pred_2_sin = reverse_shuffle(input_x, pred_2_sin, index_list)

    plotter.plot_line(x, pred_1_sin, "4 nodes")
    plotter.plot_line(x, pred_2_sin, "12 nodes")
    '''
    plotter.plot_line(input_x, pred_3_sin, "8 nodes")
    '''
    plt.legend()
    plt.show()
   


# ------------- HELPERS ---------------

def sin2x(x):
    return np.sin(2 * x)


def box2x(x):
    return 1 if np.sin(2 * x) >= 0 else -1


def phi_i(y_coor, mu, x_coor):
    point = np.array([x_coor[0], y_coor])
    phi = np.exp((- (np.linalg.norm(point - mu[:2]) ** 2) / (2 * mu[2])))
    return phi

def shuffle_data(input_x_vec, targets):
    indices = np.arange(len(targets))

    p = np.random.permutation(len(targets))
    x = input_x_vec[p]
    y = targets[p]
    indices = indices[p]
    return x, y, indices

def reverse_shuffle(y, index_list):
    ordered_y = np.zeros(len(y))
    test = np.zeros(len(y))
    for i in range(len(y)):
        index = index_list[i]
        np.put(test, index, index)
        np.put(ordered_y, index, y[i])
    return ordered_y


def main(task):
    if task == 1:
        # Task 1: Batch mode training using least squares - supervised learning of network weights
        print("----------------\n--- Task 3.1 ---\n---------------- ")
        task1()
    elif task == 2:
        # Task 2: Regression with noise
        print("----------------\n--- Task 3.2 ---\n---------------- ")
        task2()
    else:
        print("----------------\n--- Task 3.3 ---\n---------------- ")
        return 0

    # Task 3: Competitive learning (CL) to initialise RBF units
    return None

test_8_nodes()
#main(2)
