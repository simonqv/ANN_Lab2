from matplotlib import pyplot as plt
import numpy as np

X_AXIS = x = np.arange(0, 2 * np.pi, 0.1)


def sin_and_box(col, train_sin, train_box, test_sin, test_box):
    plt.plot(col, train_sin, label="Training Sin", c="b")
    plt.plot(col + 0.05, test_sin, label="Testing Sin", ls="--", c="r")
    plt.plot(col, train_box, label="Training Box", c="g")
    plt.plot(col + 0.05, test_box, label="Testing Box", ls="--", c="grey")


def plot_line(x_axis, y, lab=None, line_type="-"):
    if lab is None:
        plt.plot(x_axis, y)
    else:
        plt.plot(x_axis, y, label=lab, ls=line_type)


def points(m1, m2, m3, p1=False, p2=False, p3=False):
    """
    Beskrivning
    :param m1:
    :param m2:
    :param m3:
    :param p1:
    :param p2:
    :param p3:
    :return: geometrisk
    """
    if p1:
        p1x = []
        p1y = []
        for point in m1:
            p1x.append(point[0])
            p1y.append(point[1])
            # plt.scatter(point[0], point[1], c="b")
        plt.scatter(p1x, p1y, c="b", label="4 Units")
    if p2:
        p2x = []
        p2y = []
        for point in m2:
            p2x.append(point[0])
            p2y.append(point[1])
            # plt.scatter(point[0], point[1], marker="x", c="g")
        plt.scatter(p2x, p2y, marker="x", c="c", label="12 Units")
    if p3:
        p3x = []
        p3y = []
        for point in m3:
            p3x.append(point[0])
            p3y.append(point[1])
            # plt.scatter(point[0], point[1], marker=".", c="purple")
        plt.scatter(p3x, p3y, marker=".", c="black", label="20 Units")

def dead_vs_share(init_nodes, dead_nodes, nodes, y):
    plt.scatter(init_nodes[:,0], init_nodes[:,1], marker="x", s=50, label="Initial placement", c="#03C04A")
    plt.scatter(dead_nodes[:,0], dead_nodes[:,1], marker="o",label="Final placement dead nodes", color="#FF007F")
    plt.scatter(nodes[:,0], nodes[:,1], marker="o",label="Final placement", color="#81007F")

    plot_line(X_AXIS, y, "sin(2x)")
    plt.title("Movement of RBF nodes")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def noise(init_nodes, dead_nodes, dead_nodes_noise, train_set, train_noisy):
    plt.figure(2)
    plt.scatter(init_nodes[:,0], init_nodes[:,1], marker="x", s=50, label="Initial placement", c="#03C04A")
    plt.scatter(dead_nodes[:,0], dead_nodes[:,1], marker="o",label="Final placement wo/ noise", color="#FF007F")
    plt.scatter(dead_nodes_noise[:,0], dead_nodes_noise[:,1], marker="o",label="Final placement w/ noise", color="#81007F")

    plot_line(X_AXIS, train_set, "sin(2x)")
    plot_line(X_AXIS, train_noisy, "sin(2x) w/ noise")
    plt.title("Placements of RBF nodes\n(No dead units strategy)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
