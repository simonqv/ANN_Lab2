
from matplotlib import pyplot as plt
import numpy as np


EPOCHS = 20
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


def read_data_strings(file_path, dim = (-1, 1), names=False):
    """
    Reads a file with strings and converts to a vector with the strings in order. 
    File contains strings followed by newlines
    :param file_path: path to the data file
    """
    with open(file_path, 'r', encoding='UTF-8') as file:
        data = file.readlines()
    if not names:
        string_arr = np.array([line.strip().replace('\'', '').replace(';', '').replace(',', '').split() for line in data])
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


def update_winner_and_neighbours(attributes, weights, ind, epoch, task2):
    if not task2:
        for i in range(ind - EPOCHS + epoch , ind + EPOCHS - epoch):
            if i >= 0 and i < len(weights):
                diff = attributes - weights[i]
                weights[i] += ETA * diff
    else: 
        if epoch < 10:
            for j in range(ind - 2, ind + 3):
                diff = attributes - weights[j % len(weights)]
                weights[j % len(weights)] += ETA * diff
        elif 10 <= epoch < 15:
            for j in range(ind - 1, ind + 2):
                diff = attributes - weights[j % len(weights)]
                weights[j % len(weights)] += ETA * diff
        else:
            diff = attributes - weights[j % len(weights)]
            weights[ind % len(weights)] += ETA * diff
    
    return weights


def SOM_alg(data, weights, task2=False):
    for epoch in range(EPOCHS):
        for i, row in enumerate(data):
            attributes = row
            distances = np.linalg.norm(weights - attributes, axis=1)
            min_dist_ind = np.argmin(distances)

            weights = update_winner_and_neighbours(attributes, weights, min_dist_ind, epoch, task2)
    return weights


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


def task1():
    SOM_dimensions = [100, 84]

    # Get data
    animal_data, animal_names, animal_attributes = read_all_input_task1()
    
    # Make init weights
    init_w = np.random.rand(SOM_dimensions[0], SOM_dimensions[1])
    weights = SOM_alg(animal_data, init_w.copy())

    print_animals(animal_names, animal_data, weights)


def task2():
    SOM_dimensions = [10, 2]

    data = read_all_input_task2()

    init_w = np.random.rand(SOM_dimensions[0], SOM_dimensions[1])

    w = SOM_alg(data, init_w.copy())


    # plt.scatter(init_w[:,0], init_w[:,1], c="red", alpha=0.1)
    plt.scatter(w[:,0], w[:,1],label="SOM nodes")
    plt.scatter(data[:,0], data[:,1], c="black", marker="s", label="Cities")
    plt.plot(w[:,0], w[:,1], label="Route")
    plt.legend()
    plt.title("Cyclic Tour Between Cities")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.show()


def task3():
    SOM_dimensions = [10, 10]

    party, sex, district, names, votes = read_all_input_task3()

    init_w = np.random.rand(SOM_dimensions[0], SOM_dimensions[1])


def main():
    # task1()
    # task2()
    task3()

main()