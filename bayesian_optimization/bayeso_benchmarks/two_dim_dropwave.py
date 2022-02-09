#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 9, 2021
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    y = -1.0 * (1 + np.cos(12.0 * np.sqrt(bx[0]**2 + bx[1]**2))) / (0.5 * (bx[0]**2 + bx[1]**2) + 2.0)
    return y


class DropWave(Function):
    def __init__(self):
        dim_bx = 2
        bounds = np.array([
            [-5.12, 5.12],
            [-5.12, 5.12],
        ])
        global_minimizers = np.array([
            [0.0, 0.0],
        ])
        global_minimum = -1.0
        function = lambda bx: fun_target(bx, dim_bx)

        Function.__init__(self, dim_bx, bounds, global_minimizers, global_minimum, function)
