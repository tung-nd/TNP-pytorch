#
# author: Jungtaek Kim (jtkim@postech.ac.kr)
# last updated: February 8, 2021
#

import numpy as np

EPSILON = 1e-4


class Function(object):
    def __init__(self, dimensionality, bounds, global_minimizers, global_minimum, function, dim_problem=None, seed=None):
        assert isinstance(dimensionality, int) or dimensionality is np.inf
        assert isinstance(bounds, np.ndarray)
        assert isinstance(global_minimizers, np.ndarray)
        assert isinstance(global_minimum, float)
        assert callable(function)
        assert isinstance(dim_problem, int) or dim_problem is None
        assert len(bounds.shape) == 2
        assert bounds.shape[1] == 2
        assert (bounds[:, 0] <= bounds[:, 1]).all()

        self._dimensionality = dimensionality
        self._bounds = bounds
        self._global_minimizers = global_minimizers
        self._global_minimum = global_minimum
        self._function = function

        self.dim_problem = dim_problem
        self.random_state = np.random.RandomState(seed)

        # self.validate_properties()

    @property
    def dimensionality(self):
        return self._dimensionality

    @property
    def bounds(self):
        return self._bounds

    @property
    def global_minimizers(self):
        return self._global_minimizers

    @property
    def global_minimum(self):
        return self._global_minimum

    def get_bounds(self):
        if self.dimensionality is np.inf:
            return np.array(list(self.bounds) * self.dim_problem)
        else:
            return self.bounds

    def get_global_minimizers(self):
        if self.dimensionality is np.inf:
            global_minimizers = self.global_minimizers
            for _ in range(1, self.dim_problem):
                global_minimizers = np.concatenate((global_minimizers, self.global_minimizers), axis=1)

            return global_minimizers
        else:
            return self.global_minimizers

    def function(self, bx):
        if self.dimensionality is np.inf:
            assert self.dim_problem is bx.shape[0]
        else:
            assert self.dimensionality is bx.shape[0]

        return self._function(bx)

    def _output(self, X):
        assert isinstance(X, np.ndarray)

        if len(X.shape) == 2:
            list_results = [self.function(bx) for bx in X]
        else:
            list_results = [self.function(X)]
            
        by = np.array(list_results)
        return by

    def output(self, X):
        by = self._output(X)
        Y = np.expand_dims(by, axis=1)

        assert len(Y.shape) == 2
        assert Y.shape[1] == 1
        return Y

    def output_constant_noise(self, X, scale_noise=0.01):
        assert isinstance(scale_noise, float)

        by = self._output(X)
        by += scale_noise

        Y = np.expand_dims(by, axis=1)

        assert len(Y.shape) == 2
        assert Y.shape[1] == 1
        return Y

    def output_gaussian_noise(self, X, scale_noise=0.01):
        assert isinstance(scale_noise, float)

        by = self._output(X)
        by += scale_noise * self.random_state.randn(by.shape[0])

        Y = np.expand_dims(by, axis=1)

        assert len(Y.shape) == 2
        assert Y.shape[1] == 1
        return Y

    def output_sparse_gaussian_noise(self, X, scale_noise=0.1, sparsity=0.01):
        assert isinstance(scale_noise, float)
        assert isinstance(sparsity, float)
        assert sparsity >= 0.0 and sparsity <= 1.0
        assert sparsity < 0.5

        by = self._output(X)

        if len(X.shape) == 2:
            num_X = X.shape[0]
        else:
            num_X = 1
            
        noise = self.random_state.randn(num_X)
        mask = self.random_state.uniform(low=0.0, high=1.0, size=num_X) < sparsity
        noise *= mask.astype(np.float)
        by += scale_noise * noise

        Y = np.expand_dims(by, axis=1)

        assert len(Y.shape) == 2
        assert Y.shape[1] == 1
        return Y

    def validate_properties(self):
        shape_bounds = self.get_bounds().shape

        global_minimizers = self.get_global_minimizers()
        shape_global_minimizers = global_minimizers.shape

        assert len(shape_bounds) == 2
        assert shape_bounds[1] == 2
        assert len(shape_global_minimizers) == 2
        assert np.all(np.abs(self.output(global_minimizers) - self.global_minimum) < EPSILON)

        if self.dimensionality is np.inf:
            assert shape_bounds[0] == shape_global_minimizers[1]
        else:
            assert self.dimensionality == shape_bounds[0] == shape_global_minimizers[1]

    def get_grids(self, num_grids):
        assert isinstance(num_grids, int)

        list_grids = []
        for bound in self.get_bounds():
            list_grids.append(np.linspace(bound[0], bound[1], num_grids))
        list_grids_mesh = list(np.meshgrid(*list_grids))
        list_grids = []
        for elem in list_grids_mesh:
            list_grids.append(elem.flatten(order='C'))
        grids = np.vstack(tuple(list_grids))
        grids = grids.T
        return grids
