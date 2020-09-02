import matplotlib.pyplot as plt
import numpy as np
import math


def abc_to_rgba(A=0.0, B=0.0, C=0.0):
    """ Map values A, B, C (all in domain [0,1]) to
    suitable red, green, blue values."""
    c = np.array([min(1., max(0., 1 - (B + C))),
                  min(1., max(0., 1 - (A + C))),
                  min(1., max(0., 1 - (A + B)))])
    if np.max(c) != 0:
        c /= np.max(c)
    alpha = (A + B + C)
    return tuple(c) + (alpha,)  # / np.max(c)


# noinspection PyShadowingNames
def plot_legend_original(ax, labels=None, fontsize=None):
    """ Plots a legend for the colour scheme
    given by abc_to_rgb. Includes some code adapted
    from http://stackoverflow.com/a/6076050/637562"""

    # Basis vectors for triangle
    if labels is None:
        labels = ['A', 'B', 'C']
    basis = np.array([[0.0, 1.0], [-1.5 / np.sqrt(3), -0.5], [1.5 / np.sqrt(3), -0.5]])
    basis_fill = np.array([basis[0] + 0.1 * (basis[1] + basis[2]), basis[1] + 0.1 * (basis[0] + basis[2]),
                           basis[2] + 0.1 * (basis[0] + basis[1])])

    # Plot points
    a, b, c = np.mgrid[0.0:1.0:50j, 0.0:1.0:50j, 0.0:1.0:50j]
    a, b, c = a.flatten(), b.flatten(), c.flatten()

    abc = np.dstack((a, b, c))[0]
    # abc = np.array(list(filter(lambda x: x[0]+x[1]+x[2]==1, abc))) # remove points outside triangle
    abc = np.array(list(map(lambda x: x / sum(x), abc)))  # or just make sure points lie inside triangle ...

    data = np.dot(abc, basis_fill)
    colours = [abc_to_rgba(A=point[0], B=point[1], C=point[2]) for point in abc]

    ax.scatter(data[:, 0], data[:, 1], marker='o', edgecolors='none', facecolors=colours, s=3)

    # Plot triangle

    ax.plot([basis[_, 0] for _ in [0, 1, 2, 0]], [basis[_, 1] for _ in [0, 1, 2, 0]],
            **{'color': 'black', 'linewidth': 1})

    # Plot labels at vertices
    offset = 0.5
    ax.text(basis[0, 0] * (1 + offset), basis[0, 1] * (1 + offset), labels[0], horizontalalignment='center',
            verticalalignment='center', fontsize=fontsize)
    ax.text(basis[1, 0] * (1 + offset), basis[1, 1] * (1 + offset), labels[1], horizontalalignment='center',
            verticalalignment='center', fontsize=fontsize)
    ax.text(basis[2, 0] * (1 + offset), basis[2, 1] * (1 + offset), labels[2], horizontalalignment='center',
            verticalalignment='center', fontsize=fontsize)

    ax.set_frame_on(False)
    ax.set_xticks(())
    ax.set_yticks(())


from matplotlib.patches import Polygon


def plot_legend(ax, labels=None, fontsize=None):
    if labels is None:
        labels = ['A', 'B', 'C']
    border = 0  # 0.05
    up_triangle = np.array([[0.0, np.sqrt(3) + border], [-1. - border, 0.], [1. + border, 0.]])
    down_triangle = np.array([[0.0, 0.], [-1. - border, np.sqrt(3) + border], [1. + border, np.sqrt(3) + border]])

    N = 50
    xa, ya = 0, np.sqrt(3)
    xb, yb = -N, -(N - 1) * np.sqrt(3.)
    xc, yc = N, -(N - 1) * np.sqrt(3.)

    for i in np.arange(N):
        for j in np.arange(2 * i + 1):
            pos = np.array([j - i, -np.sqrt(3) * i])
            if j % 2:
                tri = pos + down_triangle
            else:
                tri = pos + up_triangle

            x, y = pos

            a = ((yb - yc) * (x - xc) + (xc - xb) * (y - yc)) / ((yb - yc) * (xa - xc) + (xc - xb) * (ya - yc))
            b = ((yc - ya) * (x - xc) + (xa - xc) * (y - yc)) / ((yb - yc) * (xa - xc) + (xc - xb) * (ya - yc))
            c = 1 - a - b

            c = abc_to_rgba(a, b, c)

            p = Polygon(tri, closed=True, facecolor=c, edgecolor=c, rasterized=False, lw=0.1)
            ax.add_patch(p)

    # contour triangle
    ax.plot([xa, xb, xc, xa], [ya, yb, yc, ya], **{'color': 'black', 'linewidth': 1})

    # labels
    offset = 0.3 * N
    ax.text(xa, ya + offset, labels[0], horizontalalignment='center',
            verticalalignment='center', fontsize=fontsize)
    ax.text(xb - offset, yb - offset, labels[1], horizontalalignment='center',
            verticalalignment='top', fontsize=fontsize)
    ax.text(xc + offset, yc - offset, labels[2], horizontalalignment='center',
            verticalalignment='top', fontsize=fontsize)

    ax.set_frame_on(False)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xlim([-1.05 * N, 1.05 * N])
    ax.set_ylim([-1.05 * (N - 1) * np.sqrt(3.) - 1, 1.05 * np.sqrt(3.) + 1])


if __name__ == "__main__":
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(111, aspect='equal')
    # plot_legend(ax)
    plot_legend(ax)
    plt.show()
    # print(type(abc_to_rgb(0.2,0.2,0.6)), type(abc_to_rgb(0.2,0.2,0.6)[0]))
