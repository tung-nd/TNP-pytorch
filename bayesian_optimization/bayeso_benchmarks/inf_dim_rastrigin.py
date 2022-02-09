#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(
    bx,
    dim_bx,
    A=10.0
):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx
    assert isinstance(A, float)

    y = A * dim_bx + np.sum((bx / (2.0 / 5.12)) ** 2 - A * np.cos(2 * np.pi * bx / (2.0 / 5.12)), axis=-1)
    return y


class Rastrigin(Function):
    def __init__(self, dim_problem):
        assert isinstance(dim_problem, int)

        dim_bx = np.inf
        bounds = np.array([
            [-5.12 * (2.0 / 5.12), 5.12 * (2.0 / 5.12)],
        ])
        global_minimizers = np.array([
            [0.0],
        ])
        global_minimum = 0.0
        dim_problem = dim_problem

        function = lambda bx: fun_target(bx, dim_problem)

        Function.__init__(self, dim_bx, bounds, global_minimizers, global_minimum, function, dim_problem=dim_problem)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    func = Rastrigin(dim_problem=2)
    lb, ub = func.get_bounds().transpose()
    # lb, ub = np.where(lb < -2, -2, lb), np.where(ub > 2, 2, ub)

    x1 = np.linspace(lb[0], ub[0], 50)
    x2 = np.linspace(lb[1], ub[1], 50)
    x1, x2 = np.meshgrid(x1, x2)
    pts = np.column_stack((x1.ravel(), x2.ravel()))
    func_val = func.output(pts)

    fig = plt.figure(figsize=(25, 25))

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(x1, x2, func_val.reshape(x1.shape))
    print(func.output(np.zeros((2, 2))))
    plt.show()
