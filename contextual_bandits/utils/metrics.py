import torch

from torch.distributions import Normal
from utils.misc import stack


def compute_nll(mu, sigma, y, ws=None, eps=1e-3, mask=None):
    if mask is None:
        mask = torch.ones(y.shape, dtype=torch.float32).to(mu.device)
    Ns = mu.size(0)
    sigma = sigma + eps
    py = Normal(mu, sigma)  # [Ns,B,N,Dy]
    if y.dim() < 4:
        y = torch.stack([y] * Ns, 0)  # [Ns,B,N,Dy]
    ll = (py.log_prob(y) * mask).sum(-1)  # [Ns,B,N]

    if ws is not None:
        Nbs = ws.size(2)
        ll = torch.stack([ll] * Nbs, 2)  # [Ns,B,Nbs,N]
        ll = (ll * ws).mean(2)  # [Ns,B,N]

    return - ll  # [Ns,B,N]


def compute_beta_nll(mu, sigma, y, ws=None, beta=0.5, eps=1e-3, mask=None):  # mu,sigma : [Ns,B,N,Dy], y: [B,N,Dy] ws: [Ns,B,Nbs,N]
    Ns = mu.size(0)
    sigma = sigma + eps
    y = torch.stack([y] * Ns, dim=0)  # [Ns,B,N,Dy]

    if mask is None:
        mask = torch.ones(y.shape, dtype=torch.float32).to(y.device)
    ll_mu = - ((((y - mu) ** 2) / (2 * sigma ** 2)) * mask).sum(-1)  # [Ns,B,N]
    ll_sigma = - (torch.log(sigma) * mask).sum(-1)  # [Ns,B,N]

    if ws is not None:  # [Ns,B,Nbs,N]
        Nbs = ws.size(2)
        _ll_mu = torch.stack([ll_mu] * Nbs, 2)  # [Ns,B,Nbs,N]
        _ll_sigma = torch.stack([ll_sigma] * Nbs, 2)  # [Ns,B,Nbs,N]
        _ll_mu = (_ll_mu * ws).mean(2)  # [Ns,B,N]
        _ll_sigma = (_ll_sigma * ws).mean(2)  # [Ns,B,N]
        ll = 2 * beta * _ll_mu + (2 - 2 * beta) * _ll_sigma  # [Ns,B,N]
    else:
        ll = 2 * beta * ll_mu + (2 - 2 * beta) * ll_sigma  # [Ns,B,N]

    return - ll, - ll_mu, - ll_sigma  # [Ns,B,N] all


def compute_l2(y_hat, y, ws=None, mask=None):  # pred: [Ns,B,Nbs,N,Dy], y: [B,N,Dy]
    Ns = y_hat.size(0)
    Nbs = y_hat.size(2)
    y = torch.stack([torch.stack([y] * Ns, dim=0)] * Nbs, dim=2)  # [Ns,B,Nbs,N,Dy]

    if mask is None:
        mask = torch.ones(y.shape, dtype=torch.float32).to(y.device)
    else:
        mask = stack(mask, Nbs, 2)
    l2 = (((y_hat - y) ** 2) * mask).sum(-1).mean(2)  # [Ns,B,N]
    return l2  # [Ns,B,N]


def compute_rmse(mean, y, mask=None):  # mean: [Ns,B,N,Dy], y: [B,N,Dy]
    if mean.dim() == 4:
        Ns = mean.size(0)
        y = torch.stack([y] * Ns, dim=0)  # [Ns,B,N,Dy]
        if mask is None:
            mask = torch.ones(y.shape, dtype=torch.float32).to(mean.device)
        rmse = ((((mean - y) ** 2) * mask).sum(-1).mean(-1) ** 0.5).mean()
    elif mean.dim() == 3:  # CNP, CANP
        if mask is None:
            mask = torch.ones(y.shape, dtype=torch.float32).to(mean.device)
        rmse = ((((mean - y) ** 2) * mask).sum(-1).mean(-1) ** 0.5).mean()
    return rmse
