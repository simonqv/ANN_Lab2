from matplotlib import pyplot as plt


def sin_and_box(col, train_sin, train_box, test_sin, test_box):
    plt.plot(col, train_sin)
    plt.plot(col, test_sin)
    plt.plot(col, train_box)
    plt.plot(col, test_box)


def points(m1, m2, m3):
    for x in m1:
        plt.scatter(x[0], x[1], c="b")
    for x in m2:
        plt.scatter(x[0], x[1], marker="x", c="g")
    for x in m3:
        plt.scatter(x[0], x[1], marker=".", c="pink")