from matplotlib import pyplot as plt
import numpy as np
from collections import Counter
import matplotlib.colors as colors
import matplotlib

EPOCHS = 6
ETA = 0.2


def read_data_binary(file_path, dimensions):
    """
    Reads file as one long string, replaces commas etc. and makes it into specified matrix size. 
    :param file_path: path to the data file
    :param dimensions: desired dimensions of the matrix
    :return: a matrix with the read data
    """
    with open(file_path, 'r') as file:
        data = file.read()
    data = data.replace(',', '').replace("\n", '')
    data = [int(char) for char in data]
    data_arr = np.array(data)
    return data_arr[:dimensions[0] * dimensions[1]].reshape((dimensions[0], dimensions[1]))


def read_votes(file_path, dimensions):
    with open(file_path, 'r') as file:
        data = file.read()
    data = data.split(",")
    data = [float(char) for char in data]
    data_arr = np.array(data)
    return data_arr[:dimensions[0] * dimensions[1]].reshape((dimensions[0], dimensions[1]))


def read_data_strings(file_path, dim=(-1, 1), names=False):
    """
    Reads a file with strings and converts to a vector with the strings in order. 
    File contains strings followed by newlines
    :param file_path: path to the data file
    """
    with open(file_path, 'r', encoding='UTF-8') as file:
        data = file.readlines()
    if not names:
        string_arr = np.array(
            [line.strip().replace('\'', '').replace(';', '').replace(',', '').split() for line in data])
    else:
        string_arr = np.array([line.strip().replace('\'', '').replace(';', '').replace(',', '') for line in data])
    return string_arr.reshape(dim)


def read_all_input_task1():
    file_path = "data/animals.dat"
    data_dimensions = [32, 84]
    animal_data = read_data_binary(file_path, data_dimensions)

    file_path_animal_names = "data/animalnames.txt"
    file_path_animal_attributes = "data/animalattributes.txt"
    animal_names = read_data_strings(file_path_animal_names)
    animal_attributes = read_data_strings(file_path_animal_attributes)

    return animal_data, animal_names, animal_attributes


def read_all_input_task2():
    """
    cities.dat
    % Positions of the ten cities
    % Standard positions used by e.g. [Wilson and Pawley, 1988]
    """
    file_path_cities = "data/cities.dat"
    f = read_data_strings(file_path_cities, (10, 2)).astype(float)
    return f


def read_all_input_task3():
    # These are only numbers in a column

    # Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
    # Use some color scheme for these different groups
    path_mpparty = "data/mpparty.dat"
    # % Coding: Male 0, Female 1
    path_mpsex = "data/mpsex.dat"
    path_mpdistrict = "data/mpdistrict.dat"

    path_mpnames = "data/mpnamestest.txt"
    path_votes = "data/votes.dat"

    party = read_data_strings(path_mpparty, (349, 1)).astype(int)
    sex = read_data_strings(path_mpsex, (349, 1)).astype(int)
    district = read_data_strings(path_mpdistrict, (349, 1)).astype(int)

    names = read_data_strings(path_mpnames, (349, 1), True)

    votes = read_votes(path_votes, (349, 31))

    return party, sex, district, names, votes


def update_winner_and_neighbours(attributes, weights, ind, epoch, out_ind_mat, task2_flag, task3_flag):
    if not task2_flag and not task3_flag:
        for i in range(ind - EPOCHS + epoch, ind + EPOCHS - epoch):
            if 0 <= i < len(weights):
                diff = attributes - weights[i]
                weights[i] += ETA * diff
    elif task2_flag and not task3_flag:
        if epoch < 10:
            for j in range(ind - 2, ind + 3):
                diff = attributes - weights[j % len(weights)]
                weights[j % len(weights)] += ETA * diff
        elif 10 <= epoch < 15:
            for j in range(ind - 1, ind + 2):
                diff = attributes - weights[j % len(weights)]
                weights[j % len(weights)] += ETA * diff
        else:
            for j in range(ind, ind):
                diff = attributes - weights[j % len(weights)]
                weights[ind % len(weights)] += ETA * diff
    elif task3_flag and not task2_flag:
        nh_rad = EPOCHS - epoch - 1
        winner_x, winner_y = np.unravel_index(ind, (10, 10))
        inds = []
        for i in range(10):
            for j in range(10):
                if (abs(winner_x - i) + abs(winner_y - j)) <= nh_rad:
                    inds.append(out_ind_mat[i, j])

        diff = attributes - weights[inds, :]
        weights[inds, :] += ETA * diff

    return weights


def SOM_alg(data, weights, task2_flag=False, task3_flag=False):
    if task3_flag:
        out_ind_mat = np.arange(np.prod((10, 10))).reshape((10, 10))
    else:
        out_ind_mat = []

    for epoch in range(EPOCHS):
        for i, row in enumerate(data):
            attributes = row
            distances = np.linalg.norm(weights - attributes, axis=1)
            min_dist_ind = np.argmin(distances)

            weights = update_winner_and_neighbours(attributes, weights, min_dist_ind, epoch, out_ind_mat, task2_flag, task3_flag)

    predicted_output_node = [np.argmin(np.linalg.norm(weights - row, axis=1)) for row in data]
    order_inputs = np.argsort(predicted_output_node)

    return weights, np.array(predicted_output_node), order_inputs,


def print_animals(animal_names, animal_data, weights):
    pos = []
    for i, animal in enumerate(animal_names):
        attributes = animal_data[i]
        distances = np.linalg.norm(weights - attributes, axis=1)
        min_dist_ind = np.argmin(distances)
        pos.append(min_dist_ind)
    sorting_inds = np.argsort(pos)

    sorted_animals = animal_names[sorting_inds]
    for animal in sorted_animals:
        print(animal[0])
    return sorted_animals


def creat_node_matrix(attributes, votes, weights):
    node_matrix = [[] for _ in range(100)]
    for i, personal_attribute in enumerate(attributes):
        vote_vec = votes[i]
        distances = np.linalg.norm(weights - vote_vec, axis=1)
        min_dist_ind = int(np.argmin(distances))
        node_matrix[min_dist_ind].append(personal_attribute[0])

    return node_matrix


def create_mat_map(node_mat):
    map_mat = np.zeros((10, 10))
    col = 0
    r = 0
    for i, row in enumerate(node_mat):
        if i % 10 == 0 and i > 0:
            col = 0
            r += 1
        if len(row) == 0:
            most_freq = -1
        else:
            most_freq = (Counter(row)).most_common(1)[0][0]
        map_mat[r, col] = int(most_freq)
        col += 1
    return map_mat


def noisy_index(point, noise=0.4):
    return point.astype(float)+np.random.uniform(-noise, noise, size=point.shape)


def task1():
    SOM_dimensions = [100, 84]

    # Get data
    animal_data, animal_names, animal_attributes = read_all_input_task1()

    # Make init weights
    init_w = np.random.rand(SOM_dimensions[0], SOM_dimensions[1])
    weights, _, _ = SOM_alg(animal_data, init_w.copy())

    print_animals(animal_names, animal_data, weights)


def task2():
    SOM_dimensions = [10, 2]

    data = read_all_input_task2()

    init_w = np.random.rand(SOM_dimensions[0], SOM_dimensions[1])

    w, _, _ = SOM_alg(data, init_w.copy())

    # plt.scatter(init_w[:,0], init_w[:,1], c="red", alpha=0.1)
    plt.scatter(w[:, 0], w[:, 1], label="SOM nodes")
    plt.scatter(data[:, 0], data[:, 1], c="black", marker="s", label="Cities")
    plt.plot(w[:, 0], w[:, 1], label="Route")
    plt.legend()
    plt.title("Cyclic Tour Between Cities")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.show()


def task3():
    # number of weight vectors should be the number of features times the number of clusters (100)
    # total number of weight vectors are 31 x 100

    # SOM_output_grid_dimensions = [10, 10]

    weight_dimensions = [100, 31]

    party, sex, district, names, votes = read_all_input_task3()

    init_w = np.random.rand(weight_dimensions[0], weight_dimensions[1])

    # TODO: REPEAT FOR ALL (sex, district, (names?))
    weights, predicted_output_node, order_inputs, = SOM_alg(votes, init_w.copy())
    print(weights.shape)
    node_matrix_all_party = creat_node_matrix(party, votes, weights)
    mat_map_party = create_mat_map(node_matrix_all_party)
    print(mat_map_party)

    # TODO: make the colors correspond to parties!!
    # Coding: -1 no votes, 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'

    '''
        Centerpartiet	Skogsgrön	#009933
      
        Kristdemokraterna	Mörkblå	#000077
        Liberalerna	Marinblå	#006AB3
        Miljöpartiet	Maskrosgrön	#83CF39
        Moderaterna	Ljusblå	#52BDEC
      
        Socialdemokraterna	Röd	#E8112d
        
        Vänsterpartiet	Mörkröd	#DA291C
    '''
    colors_list = ["#FFFFFF", "#000000", "#52BDEC", "#006AB3", "#E8112d", "#870000", "#83CF39", "#000077", "#009933"]
    color_map = colors.ListedColormap(colors_list)

    plt.figure(1)
    fig, ax = plt.subplots()
    cax = ax.imshow(mat_map_party, cmap=color_map, alpha=0.8) 
    ax.set_title("Majority Party for each Node")
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1, 2, 3, 4, 5, 6, 7], orientation="horizontal")
    cbar.ax.set_xticklabels(["No win", "No party", "M", "FP", "S", "V", "MP", "KD", "C"])
    ax.set_yticks([])
    ax.set_xticks([])
    plt.show()
    # ---- Sex ------
    node_matrix_all_sex = creat_node_matrix(sex, votes, weights)
    mat_map_sex = create_mat_map(node_matrix_all_sex)

    plt.figure(2)
    color_map = colors.ListedColormap(colors_list[:3])
    plt.imshow(mat_map_sex, cmap=color_map, alpha=0.8)
    plt.colorbar(ticks=[-1, 0, 1]).ax.tick_params(labelsize=10)
    plt.title("Sex")



    '''
   
    # TODO: REPEAT FOR ALL (sex, district, (names?))
    weights = SOM_alg(votes, init_w.copy())
    node_matrix_all_sex = creat_node_matrix(sex, votes, weights)
    mat_map_sex = create_mat_map(node_matrix_all_sex)

    plt.figure(2)
    colors_s = 2
    plt.imshow(mat_map_sex, cmap=colors.ListedColormap(colors_list[:3]))
    plt.colorbar()
    plt.title("Sex")
    plt.show()

    '''
    
    plt.figure(3)

    mp_attr_names = ["Party", "District", "Sex"]
    # votes = np.genfromtxt('data/votes.dat', delimiter=',').reshape(349, 31)
    party = np.genfromtxt('data/mpparty.dat', comments='%', dtype=np.uint8)
    district = np.genfromtxt('data/mpdistrict.dat', comments='%', dtype=np.uint8)
    sex = np.genfromtxt('data/mpsex.dat', comments='%', dtype=np.uint8)
    mp_attrs = np.column_stack((party, district, sex))
    mp_attr_levels = [
        ["no party", "m", "fp", "s", "v",
         "mp", "kd", "c"],
        ["District " + str(d) for d in np.unique(district)],
        ["Male", "Female"]]
    plot_colors = [
        colors_list[1:],#["black", "#52BDEC", "#006AB3", "#E8112d", "pink", "#83CF39", "#000077", "#009933"],
        [],
        []
    ]
    # Iterate feature names
    for attr in range(len(mp_attr_names)):
        fig = plt.figure(figsize=(5, 5))

        for i, row in enumerate(votes):
            attributes = row
            distances = np.linalg.norm(weights - attributes, axis=1)
            min_dist_ind = np.argmin(distances)

        # Iterate unique values in that feature
        for val in np.unique(mp_attrs[:, attr]):
            # Find the output node assigned to each minister with that feature value
            x, y = np.unravel_index(predicted_output_node[mp_attrs[:, attr] == val], (10, 10))
            #print(x, y)
            # Plot the position in output space of each minister with that feature value
            if attr == 0:
                plt.scatter(noisy_index(x), noisy_index(y), s=20, alpha=1, label=mp_attr_levels[attr][val-1], c=plot_colors[attr][val - 1])
            else:
                plt.scatter(noisy_index(x), noisy_index(y), s=20, alpha=1, label=mp_attr_levels[attr][val-1])

        # Edit the plot
        plt.title(mp_attr_names[attr], fontsize=30, y=1.02)
        plt.xticks(np.arange(-0.5, 10 - 0.5, 1))#, labels=[])
        plt.yticks(np.arange(-0.5, 10 - 0.5, 1))#, labels=[])
        plt.grid(True)
        plt.xlim([-0.5, 10 - 0.5])
        plt.ylim([-0.5, 10 - 0.5])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
    plt.show()


def main():
    # task1()
    task2()
    # task3()


main()
