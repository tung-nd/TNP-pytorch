#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
    ])
    P = 1e-4 * np.array([
        [3689, 1170, 2673],
        [4699, 4387, 7470],
        [1091, 8732, 5547],
        [381, 5743, 8828],
    ])

    outer = 0.0
    for i_ in range(0, 4):
        inner = 0.0
        for j_ in range(0, 3):
            inner += A[i_, j_] * (bx[j_] - P[i_, j_])**2
        outer += alpha[i_] * np.exp(-1.0 * inner)

    y = -1.0 * outer
    return y


class Hartmann3D(Function):
    def __init__(self,
        bounds=np.array([
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ])
    ):
        assert isinstance(bounds, np.ndarray)
        assert len(bounds.shape) == 2
        assert bounds.shape[1] == 2

        dim_bx = 3
        assert bounds.shape[0] == dim_bx

        global_minimizers = np.array([
            [0.114614, 0.555649, 0.852547],
        ])
        global_minimum = -3.86278
        function = lambda bx: fun_target(bx, dim_bx)

        Function.__init__(self, dim_bx, bounds, global_minimizers, global_minimum, function)
