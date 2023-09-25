from matplotlib import pyplot as plt


def sin_and_box(col, train_sin, train_box, test_sin, test_box):
    plt.plot(col, train_sin)
    plt.plot(col, test_sin)
    plt.plot(col, train_box)
    plt.plot(col, test_box)

def plot_line(x_axis, y, l=None):
    if l == None:
        plt.plot(x_axis, y)
    else:
        plt.plot(x_axis, y, label=l)


def points(m1, m2, m3, p1=False, p2=False, p3=False):

    if p1:
        for x in m1:
            plt.scatter(x[0], x[1], c="b")
    if p2:
        for x in m2:
            plt.scatter(x[0], x[1], marker="x", c="g")
    if p3:
        for x in m3:
            plt.scatter(x[0], x[1], marker=".", c="pink")