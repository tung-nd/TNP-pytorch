import numpy as np
import torch

from attrdict import AttrDict
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor
from torch.distributions import Normal
from torch.nn import Module
from typing import Union


class EI(AnalyticAcquisitionFunction):
    def __init__(
            self,
            model: Module,
            observations: AttrDict,
            best_f: Union[float, Tensor],
            num_bs: int = 200,
            maximize: bool = True
    ):
        model.num_outputs = 1
        super(EI, self).__init__(model=model)

        self.obs = observations
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)
        self.num_bs = num_bs
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        self.best_f = self.best_f.to(X)

        posterior = self.model.predict(xc=self.obs.xc,
                                        yc=self.obs.yc,
                                        xt=X,
                                        num_samples=self.num_bs)
        mean, std = posterior.mean.squeeze(0), posterior.scale.squeeze(0)

        # shape: (num_samples, 1, num_points, 1)
        if mean.dim() == 4:
            var = std.pow(2).mean(dim=0) + mean.pow(2).mean(dim=0) - mean.mean(dim=0).pow(2)
            std = var.sqrt().squeeze(0)
            mean = mean.mean(dim=0).squeeze(0)

        batch_shape = mean.shape[:-2] if mean.dim() >= X.dim() else X.shape[:-2]
        mean = mean.view(batch_shape)
        std = std.clamp_min(np.sqrt(1e-9)).view(batch_shape)
        u = (mean - self.best_f.expand_as(mean)) / std
        if not self.maximize:
            u = -u
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = std * (updf + u * ucdf)
        return ei


class UCB(AnalyticAcquisitionFunction):
    def __init__(
            self,
            model: Module,
            observations: AttrDict,
            beta: Union[float, Tensor],
            num_bs: int = 200,
            maximize: bool = True
    ):
        model.num_outputs = 1
        super(UCB, self).__init__(model=model)

        self.obs = observations
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)
        self.num_bs = num_bs
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor, return_mean=False) -> Tensor:
        self.beta = self.beta.to(X)

        posterior = self.model.predict(xc=self.obs.xc,
                                        yc=self.obs.yc,
                                        xt=X,
                                        num_samples=self.num_bs)
        mean, std = posterior.mean.squeeze(0), posterior.scale.squeeze(0)

        # shape: (num_samples, 1, num_points, 1)
        if mean.dim() == 4:
            var = std.pow(2).mean(dim=0) + mean.pow(2).mean(dim=0) - mean.mean(dim=0).pow(2)
            std = var.sqrt().squeeze(0)
            mean = mean.mean(dim=0).squeeze(0)

        batch_shape = X.shape[:-2]
        mean = mean.view(batch_shape)
        std = std.view(batch_shape)
        delta = self.beta.expand_as(mean).sqrt() * std
        if return_mean:
            return mean
        else:
            if self.maximize:
                return mean + delta
            else:
                return -mean + delta
