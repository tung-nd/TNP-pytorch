import torch
from torch.distributions import MultivariateNormal, StudentT
from attrdict import AttrDict
import math


__all__ = ["GPPriorSampler", 'GPSampler', 'RBFKernel', 'PeriodicKernel', 'Matern52Kernel']


class GPPriorSampler(object):
    """
    Bayesian Optimization에서 이용
    """
    def __init__(self, kernel, t_noise=None):
        self.kernel = kernel
        self.t_noise = t_noise

    # bx: 1 * num_points * 1
    def sample(self, x, device):
        # 1 * num_points * num_points
        cov = self.kernel(x)
        mean = torch.zeros(1, x.shape[1], device=device)

        y = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)

        if self.t_noise is not None:
            y += self.t_noise * StudentT(2.1).rsample(y.shape).to(device)

        return y


class GPSampler(object):
    def __init__(self, kernel, t_noise=None, seed=None):
        self.kernel = kernel
        self.t_noise = t_noise
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        self.seed = seed

    def sample(self,
            batch_size=16,
            num_ctx=None,
            num_tar=None,
            max_num_points=50,
            x_range=(-2, 2),
            device='cpu'):

        batch = AttrDict()
        num_ctx = num_ctx or torch.randint(low=3, high=max_num_points-3, size=[1]).item()  # Nc
        num_tar = num_tar or torch.randint(low=3, high=max_num_points-num_ctx, size=[1]).item()  # Nt

        num_points = num_ctx + num_tar  # N = Nc + Nt
        batch.x = x_range[0] + (x_range[1] - x_range[0]) \
                * torch.rand([batch_size, num_points, 1], device=device)  # [B,N,Dx=1]
        batch.xc = batch.x[:,:num_ctx]  # [B,Nc,1]
        batch.xt = batch.x[:,num_ctx:]  # [B,Nt,1]

        # batch_size * num_points * num_points
        cov = self.kernel(batch.x)  # [B,N,N]
        mean = torch.zeros(batch_size, num_points, device=device)  # [B,N]
        batch.y = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)  # [B,N,Dy=1]
        batch.yc = batch.y[:,:num_ctx]  # [B,Nc,1]
        batch.yt = batch.y[:,num_ctx:]  # [B,Nt,1]

        if self.t_noise is not None:
            if self.t_noise == -1:
                t_noise = 0.15 * torch.rand(batch.y.shape).to(device)  # [B,N,1]
            else:
                t_noise = self.t_noise
            batch.y += t_noise * StudentT(2.1).rsample(batch.y.shape).to(device)
        return batch
        # {"x": [B,N,1], "xc": [B,Nc,1], "xt": [B,Nt,1],
        #  "y": [B,N,1], "yc": [B,Nt,1], "yt": [B,Nt,1]}


class RBFKernel(object):
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim  [B,N,Dx=1]
    def __call__(self, x):
        length = 0.1 + (self.max_length-0.1) \
                * torch.rand([x.shape[0], 1, 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)

        # batch_size * num_points * num_points * dim  [B,N,N,1]
        dist = (x.unsqueeze(-2) - x.unsqueeze(-3))/length

        # batch_size * num_points * num_points  [B,N,N]
        cov = scale.pow(2) * torch.exp(-0.5 * dist.pow(2).sum(-1)) \
                + self.sigma_eps**2 * torch.eye(x.shape[-2]).to(x.device)

        return cov  # [B,N,N]


class Matern52Kernel(object):
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim
    def __call__(self, x):
        length = 0.1 + (self.max_length-0.1) \
                * torch.rand([x.shape[0], 1, 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)

        # batch_size * num_points * num_points
        dist = torch.norm((x.unsqueeze(-2) - x.unsqueeze(-3))/length, dim=-1)

        cov = scale.pow(2)*(1 + math.sqrt(5.0)*dist + 5.0*dist.pow(2)/3.0) \
                * torch.exp(-math.sqrt(5.0) * dist) \
                + self.sigma_eps**2 * torch.eye(x.shape[-2]).to(x.device)

        return cov


class PeriodicKernel(object):
    def __init__(self, sigma_eps=2e-2, max_length=0.6, max_scale=1.0):
        #self.p = p
        self.sigma_eps = sigma_eps
        self.max_length = max_length
        self.max_scale = max_scale

    # x: batch_size * num_points * dim
    def __call__(self, x):
        p = 0.1 + 0.4*torch.rand([x.shape[0], 1, 1], device=x.device)
        length = 0.1 + (self.max_length-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)
        scale = 0.1 + (self.max_scale-0.1) \
                * torch.rand([x.shape[0], 1, 1], device=x.device)

        dist = x.unsqueeze(-2) - x.unsqueeze(-3)
        cov = scale.pow(2) * torch.exp(
            - 2*(torch.sin(math.pi*dist.abs().sum(-1)/p)/length).pow(2)) \
                + self.sigma_eps**2 * torch.eye(x.shape[-2]).to(x.device)

        return cov
