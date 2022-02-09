#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(
        bx,
        dim_bx,
        a=20.0,
        b=0.2,
        c=2.0 * np.pi
):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx
    assert isinstance(a, float)
    assert isinstance(b, float)
    assert isinstance(c, float)

    y = -a * np.exp(-b * np.linalg.norm(bx, ord=2, axis=0) * np.sqrt(1.0 / dim_bx)) - np.exp(
        1.0 / dim_bx * np.sum(np.cos(c * bx), axis=0)) + a + np.exp(1.0)
    return y


class Ackley(Function):
    def __init__(self, dim_problem):
        assert isinstance(dim_problem, int)

        dim_bx = np.inf
        bounds = np.array([
            [-32.768, 32.768],
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

    func = Ackley(dim_problem=2)
    lb, ub = func.get_bounds().transpose()
    lb, ub = np.where(lb < -2, -2, lb), np.where(ub > 2, 2, ub)

    x1 = np.linspace(lb[0], ub[0], 100)
    x2 = np.linspace(lb[1], ub[1], 100)
    x1, x2 = np.meshgrid(x1, x2)
    pts = np.column_stack((x1.ravel(), x2.ravel()))
    y = func.output(pts)

    contour = plt.contourf(x1, x2, y.reshape(x1.shape), 50, cmap='RdGy')
    # contour = plt.contourf(x1, x2, y.reshape(x1.shape), 50)
    plt.imshow(y.reshape(x1.shape), extent=[lb[0], ub[0], lb[1], ub[1]], origin='lower',
               cmap='RdGy')
    plt.colorbar(contour)
    plt.show()
