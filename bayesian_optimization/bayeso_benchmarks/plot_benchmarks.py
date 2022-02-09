import numpy as np
import os
import matplotlib.pyplot as plt


def plot_1d(obj_fun,
    str_fun,
    str_x_axis=r'$x$',
    str_y_axis=r'$f(x)$',
    str_figures='../figures',
):
    print(str_fun)
    bounds = obj_fun.get_bounds()
    print(bounds)
    assert bounds.shape[0] == 1

    X = np.linspace(bounds[0, 0], bounds[0, 1], 1000)
    Y = obj_fun.output(X[..., np.newaxis]).flatten()

    assert len(X.shape) == 1
    assert len(Y.shape) == 1
    assert X.shape[0] == Y.shape[0]

    plt.rc('text', usetex=True)

    _ = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    ax.plot(X, Y,
        linewidth=4,
        marker='None')

    ax.set_xlabel(str_x_axis, fontsize=36)
    ax.set_ylabel(str_y_axis, fontsize=36)
    ax.tick_params(labelsize=24)

    ax.set_xlim([np.min(X), np.max(X)])
    ax.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(str_figures, str_fun + '.pdf'),
        format='pdf',
        transparent=True,
        bbox_inches='tight')

    plt.show()

def plot_2d(obj_fun,
    str_fun,
    str_x1_axis=r'$x_1$',
    str_x2_axis=r'$x_2$',
    str_y_axis=r'$f(\mathbf{x})$',
    str_figures='../figures',
):
    print(str_fun)
    bounds = obj_fun.get_bounds()
    print(bounds)
    assert bounds.shape[0] == 2

    X1 = np.linspace(bounds[0, 0], bounds[0, 1], 200)
    X2 = np.linspace(bounds[1, 0], bounds[1, 1], 200)
    X1, X2 = np.meshgrid(X1, X2)
    X = np.concatenate((X1[..., np.newaxis], X2[..., np.newaxis]), axis=2)
    X = np.reshape(X, (X.shape[0] * X.shape[1], X.shape[2]))

    Y = obj_fun.output(X).flatten()

    assert len(X.shape) == 2
    assert len(Y.shape) == 1
    assert X.shape[0] == Y.shape[0]

    Y = np.reshape(Y, (X1.shape[0], X2.shape[0]))

    plt.rc('text', usetex=True)

    _ = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection='3d')

    surf = ax.plot_surface(X1, X2, Y,
        cmap='coolwarm',
        linewidth=0)

    ax.set_xlabel(str_x1_axis, fontsize=24, labelpad=10)
    ax.set_ylabel(str_x2_axis, fontsize=24, labelpad=10)
    ax.set_zlabel(str_y_axis, fontsize=24, labelpad=10)
    ax.tick_params(labelsize=16)

    ax.set_xlim([np.min(X1), np.max(X1)])
    ax.set_ylim([np.min(X2), np.max(X2)])
    ax.grid()

    cbar = plt.colorbar(surf,
        shrink=0.6,
        aspect=12,
        pad=0.15,
    )
    cbar.ax.tick_params(labelsize=16)

    if np.max(Y) > 1000:
        plt.ticklabel_format(axis='z', style='sci', scilimits=(0, 0), useMathText=True)
        ax.zaxis.get_offset_text().set_fontsize(14)

    plt.tight_layout()
    plt.savefig(os.path.join(str_figures, str_fun + '.pdf'),
        format='pdf',
        transparent=True,
        bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    # one dim.

    from inf_dim_ackley import Ackley as target_class
    obj_fun = target_class(1)
    plot_1d(obj_fun, 'ackley_1d')

    from inf_dim_cosines import Cosines as target_class
    obj_fun = target_class(1)
    plot_1d(obj_fun, 'cosines_1d')


    # two dim.
    from two_dim_dropwave import DropWave as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'dropwave_2d')

    from two_dim_goldsteinprice import GoldsteinPrice as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'goldsteinprice_2d')

    from two_dim_michalewicz import Michalewicz as target_class
    obj_fun = target_class()
    plot_2d(obj_fun, 'michalewicz_2d')

    from inf_dim_ackley import Ackley as target_class
    obj_fun = target_class(2)
    plot_2d(obj_fun, 'ackley_2d')

    from inf_dim_cosines import Cosines as target_class
    obj_fun = target_class(2)
    plot_2d(obj_fun, 'cosines_2d')

