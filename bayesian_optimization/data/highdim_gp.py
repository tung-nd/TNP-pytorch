import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import warnings
sns.set()
warnings.filterwarnings('ignore')

from attrdict import AttrDict
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.priors import UniformPrior
from mpl_toolkits import mplot3d
from typing import Union, List, Tuple


class GaussianProcess(ExactGP):
    def __init__(self, x, y, likelihood, device):
        super(GaussianProcess, self).__init__(x, y, likelihood)
        self.mean_module = ConstantMean()

        self.length_prior = UniformPrior(0.1, 1.0)
        self.scale_prior = UniformPrior(0.1, 1.0)

        self.covar_module = ScaleKernel(
            RBFKernel(lengthscale_prior=self.length_prior),
            outputscale_prior=self.scale_prior
        )
        self.device = device

    def forward(self, x, verbose=False, random_parameter=True):
        # Sample lengthscale and outputscale randomly
        if random_parameter:
            self.covar_module.base_kernel.lengthscale = self.length_prior.rsample().to(self.device)
            self.covar_module.outputscale = self.scale_prior.rsample().to(self.device)

        if verbose:
            print(f'Actual length scale: {self.covar_module.base_kernel.lengthscale}')
            print(f'Actual output scale: {self.covar_module.outputscale}')
            print('=' * 70)

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPSampler:
    def __init__(
            self,
            dimension: int = 5,
            device: torch.device = torch.device('cpu'),
            seed: int = None
    ):
        # initialize likelihood and gp
        likelihood = GaussianLikelihood().to(device)
        self.gp = GaussianProcess(None, None, likelihood=likelihood, device=device).to(device)
        self.gp.eval()

        self.dim = dimension
        self.device = device
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

    def __call__(
            self,
            batch_size: int = 16,
            num_ctx: int = None,
            num_tar: int = None,
            max_num_points: int = 512,
            min_num_points: int = 128,
            x_range: Union[List, Tuple] = (-2, 2),
            random_parameter: bool = True
    ):
        lb, ub = x_range

        batch = AttrDict()

        num_ctx = num_ctx or torch.randint(min_num_points, max_num_points - min_num_points, size=[1]).item()
        num_tar = num_tar or torch.randint(min_num_points, max_num_points - num_ctx, size=[1]).item()

        num_points = num_ctx + num_tar
        batch.x = lb + (ub - lb) * torch.rand([batch_size, num_points, self.dim], device=self.device)
        batch.xc = batch.x[:, :num_ctx]
        batch.xt = batch.x[:, num_ctx:]

        with gpytorch.settings.prior_mode(True):
            batch.y = self.gp(batch.x,
                              verbose=False,
                              random_parameter=random_parameter).rsample().unsqueeze(-1)
            batch.yc = batch.y[:, :num_ctx]
            batch.yt = batch.y[:, num_ctx:]

        return batch


if __name__ == '__main__':
    sampler = GPSampler(dimension=2)

    fig = plt.figure(figsize=(35, 35))

    for i, p in enumerate([25, 500], 1):
        pts = sampler(num_ctx=p, num_tar=p, random_parameter=False)

        ax = fig.add_subplot(1, 2, i, projection='3d')
        ax.scatter(pts.x[0, :, 0].detach().numpy(),
                   pts.x[0, :, 1].detach().numpy(),
                   pts.y[0].detach().numpy())
    plt.show()