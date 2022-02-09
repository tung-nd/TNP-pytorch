#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx, steps, step_values):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx
    assert isinstance(steps, list)
    assert isinstance(step_values, list)
    assert len(steps) == len(step_values) + 1
    assert isinstance(steps[0], float)
    assert isinstance(step_values[0], float)

    y = None
    for ind_step in range(0, len(steps) - 1):
        if ind_step < (len(steps) - 2) and steps[ind_step] <= bx[0] and bx[0] < steps[ind_step+1]:
            y = step_values[ind_step]
            break
        elif ind_step == (len(steps) - 2) and steps[ind_step] <= bx[0] and bx[0] <= steps[ind_step+1]:
            y = step_values[ind_step]
            break

    if y is None:
        raise ValueError('Conditions for steps')
    return y


class Step(Function):
    def __init__(self,
        steps=[-10., -5., 0., 5., 10.],
        step_values=[-2., 0., 1., -1.],
    ):
        assert isinstance(steps, list)
        assert isinstance(step_values, list)
        assert len(steps) == len(step_values) + 1
        assert isinstance(steps[0], float)
        assert isinstance(step_values[0], float)
        assert np.all(np.sort(steps) == np.asarray(steps))

        dim_bx = 1
        bounds = np.array([
            [np.min(steps), np.max(steps)],
        ])
        global_minimizers = np.array([
            [steps[np.argmin(step_values)]],
        ])
        global_minimum = np.min(step_values)
        function = lambda bx: fun_target(bx, dim_bx, steps, step_values)

        Function.__init__(self, dim_bx, bounds, global_minimizers, global_minimum, function)
