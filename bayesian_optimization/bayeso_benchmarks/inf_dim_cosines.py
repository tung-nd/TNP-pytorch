#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = np.sum(np.cos(bx) * (np.abs(bx) * (0.1 / (2.0 * np.pi)) - 1.0))
    return y


class Cosines(Function):
    def __init__(self, dim_problem):
        assert isinstance(dim_problem, int)

        dim_bx = np.inf
        bounds = np.array([
            [-2.0 * np.pi, 2.0 * np.pi],
        ])
        global_minimizers = np.array([
            [0.0],
        ])
        global_minimum = -1.0 * dim_problem
        dim_problem = dim_problem

        function = lambda bx: fun_target(bx, dim_problem)

        Function.__init__(self, dim_bx, bounds, global_minimizers, global_minimum, function, dim_problem=dim_problem)
