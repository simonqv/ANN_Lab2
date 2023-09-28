from part1 import make_phi_matrix, batch_least_squares, residual_err
import numpy as np
from matplotlib import pyplot as plt

SIGMA2 = 0.5


def load_ball_data(file_path):
    data = np.loadtxt(file_path)
    # Split the data into separate arrays for each column
    columns = np.hsplit(data, 4)
    # print(columns)
    angle = columns[0].reshape(len(columns[0]), 1)
    velocity = columns[1].reshape(len(columns[1]), 1)
    distance = columns[2].reshape(len(columns[2]), 1)
    height = columns[3].reshape(len(columns[3]), 1)
    # print("SHAPEH", angle.shape)

    pattern = np.concatenate((angle, velocity), axis=1)
    target = np.concatenate((distance, height), axis=1)

    return pattern, target # angle, velocity, distance, height

"""
def generate_rand_rbfs_ball(len_data_set):
    # generate 12 random nodes
    # x values
    x = np.random.uniform(low=0, high=len_data_set, size=(len_data_set,1))
    #y values between 1- and 1
    y = np.random.uniform(-1, 1, (len_data_set, 1))
    sigmas = np.full(12, SIGMA2).reshape(-1, 1)

    nodes = np.hstack((x, y, sigmas))
    return nodes
"""


def make_rbf_matrix(pattern, n_nodes):  
    np.random.shuffle(pattern)
    selected_data = pattern[:n_nodes]

    x_vals = selected_data[:, 0]
    y_vals = selected_data[:, 1]

    sigma_arr = np.full(n_nodes, SIGMA2)

    return np.column_stack((x_vals, y_vals, sigma_arr))


def cl(nodes_in, pattern):
    nodes = nodes_in.copy()
    eta = 0.5
    epochs = 100
    
    for _ in range(epochs):
        # Pick a random data point. 
        input_vec_i = np.random.randint(0, len(pattern))
        data_point = pattern[input_vec_i]

        dists = [0 for _ in range(len(nodes))]
        for index, node in enumerate(nodes):
            # Calculate euclidean distance
            dist = np.linalg.norm(data_point - node[:2])
            dists[index] = dist
        
        winner_i = np.argmin(dists)
        winner = nodes[winner_i]

        winner_diff = np.array([data_point[0] - winner[0], data_point[1] - winner[1], 0])

        # Update position (w) 
        nodes[winner_i] += (eta * winner_diff)

    return nodes


def task3_ball(N_NODES):
    # N_NODES = 10
    # Load trainging data
    file_path = "data/ballist.dat"
    pattern, target = load_ball_data(file_path)

    # Load testing data
    file_path_test = "data/balltest.dat"
    pattern_test, target_test = load_ball_data(file_path_test)

    init_rbf_nodes = make_rbf_matrix(pattern.copy(), N_NODES)
    
    
    rbf_nodes = cl(init_rbf_nodes, pattern)
    
    """
    plt.scatter(init_rbf_nodes[:,0], init_rbf_nodes[:,1], label="Start")
    plt.scatter(rbf_nodes[:,0], rbf_nodes[:,1], label="Final")
    plt.legend()
    plt.show()
    """
    
    # phi matrix for sin
    phi_sin = make_phi_matrix(rbf_nodes, pattern[:,1], pattern[:,0], True, True)


    # weight vectors
    # maybe make two.. try one
    # TRAINING
    w_sin = batch_least_squares(phi_sin, target, task_ball=True)
    

    # test on hold out set and sum to get output
    phi_sin_test = make_phi_matrix(rbf_nodes, pattern_test[:,1], pattern_test[:,0], True, True)
    out_sin = np.dot(phi_sin_test, w_sin)
    # print(out_sin - target_test)
    
    err = residual_err(out_sin, target_test)
    print(f"Residual error with {N_NODES} RBF nodes: {err}")

    plt.figure(N_NODES)
    plt.title(f"Prediction vs Truth with {N_NODES} RBF nodes")
    for i in range(0, len(out_sin)):
        plt.plot([out_sin[i, 0], target_test[i, 0]],[out_sin[i, 1], target_test[i, 1]] , ls='--', c="grey")

    plt.scatter(out_sin[:, 0], out_sin[:, 1], label="Prediction", marker=".")
    plt.scatter(target_test[:, 0], target_test[:, 1], label="Truth", marker=".")
    plt.xlabel("Distance")
    plt.ylabel("Height")
    plt.legend()
    


def main():
    task3_ball(6)
    task3_ball(12)
    task3_ball(15)
    plt.show()

    

main()