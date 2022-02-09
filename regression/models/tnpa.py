import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from attrdict import AttrDict

from utils.misc import stack
from models.tnp import TNP


class TNPA(TNP):
    def __init__(
        self,
        dim_x,
        dim_y,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers,
        emnist=False,
        permute=False
    ):
        super(TNPA, self).__init__(
            dim_x,
            dim_y,
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            dropout,
            num_layers,
            emnist
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y*2)
        )

        self.permute = permute

    def forward(self, batch, reduce_ll=True):
        z_target = self.encode(batch, autoreg=True)
        out = self.predictor(z_target)
        mean, std = torch.chunk(out, 2, dim=-1)
        if self.emnist:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            std = torch.exp(std)

        pred_tar = Normal(mean, std)

        outs = AttrDict()
        if reduce_ll:
            outs.tar_ll = pred_tar.log_prob(batch.yt).sum(-1).mean()
        else:
            outs.tar_ll = pred_tar.log_prob(batch.yt).sum(-1)
        outs.loss = - (outs.tar_ll)

        return outs

    def permute_sample_batch(self, xt, yt, num_samples, batch_size, num_target):
        # data in each batch is permuted identically
        perm_ids = torch.rand(num_samples, num_target, device='cuda').unsqueeze(1).repeat((1, batch_size, 1))
        perm_ids = torch.argsort(perm_ids, dim=-1)
        deperm_ids = torch.argsort(perm_ids, dim=-1)
        dim_sample = torch.arange(num_samples, device='cuda').unsqueeze(-1).unsqueeze(-1).repeat((1,batch_size,num_target))
        dim_batch = torch.arange(batch_size, device='cuda').unsqueeze(0).unsqueeze(-1).repeat((num_samples,1,num_target))
        return xt[dim_sample, dim_batch, perm_ids], yt[dim_sample, dim_batch, perm_ids], dim_sample, dim_batch, deperm_ids

    def predict(self, xc, yc, xt, num_samples=50, return_samples=False):
        batch_size = xc.shape[0]
        num_target = xt.shape[1]
        
        def squeeze(x):
            return x.view(-1, x.shape[-2], x.shape[-1])
        def unsqueeze(x):
            return x.view(num_samples, batch_size, x.shape[-2], x.shape[-1])

        xc_stacked = stack(xc, num_samples)
        yc_stacked = stack(yc, num_samples)
        xt_stacked = stack(xt, num_samples)
        yt_pred = torch.zeros((batch_size, num_target, yc.shape[2]), device='cuda')
        yt_stacked = stack(yt_pred, num_samples)
        if self.permute:
            xt_stacked, yt_stacked, dim_sample, dim_batch, deperm_ids = self.permute_sample_batch(xt_stacked, yt_stacked, num_samples, batch_size, num_target)

        batch_stacked = AttrDict()
        batch_stacked.xc = squeeze(xc_stacked)
        batch_stacked.yc = squeeze(yc_stacked)
        batch_stacked.xt = squeeze(xt_stacked)
        batch_stacked.yt = squeeze(yt_stacked)

        for step in range(xt.shape[1]):
            z_target_stacked = self.encode(batch_stacked, autoreg=True)
            out = self.predictor(z_target_stacked)
            mean, std = torch.chunk(out, 2, dim=-1)
            if self.emnist:
                std = 0.05 + 0.95 * F.softplus(std)
            else:
                std = torch.exp(std)
            mean, std = unsqueeze(mean), unsqueeze(std)
            batch_stacked.yt = unsqueeze(batch_stacked.yt)
            batch_stacked.yt[:, :, step] = Normal(mean[:, :, step], std[:, :, step]).sample()
            batch_stacked.yt = squeeze(batch_stacked.yt)

        if self.permute:
            mean, std = mean[dim_sample, dim_batch, deperm_ids], std[dim_sample, dim_batch, deperm_ids]

        if return_samples:
            return unsqueeze(batch_stacked.yt)

        return Normal(mean, std)

    def sample(self, xc, yc, xt, num_samples=50):
        return self.predict(xc, xt, yt, num_samples, return_samples=True)