from matplotlib import pyplot as plt


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
        plt.scatter(p2x, p2y, marker="s", c="c", label="12 Units")
    if p3:
        p3x = []
        p3y = []
        for point in m3:
            p3x.append(point[0])
            p3y.append(point[1])
            # plt.scatter(point[0], point[1], marker=".", c="purple")
        plt.scatter(p3x, p3y, marker=".", c="black", label="20 Units")
