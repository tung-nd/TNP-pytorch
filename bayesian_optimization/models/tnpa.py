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
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y*2)
        )

    def forward(self, batch, reduce_ll=True):
        num_ctx, num_all = batch.xc.shape[1], batch.x.shape[1]

        out_encoder = self.encode(batch, autoreg=True)
        out = self.predictor(out_encoder)
        mean, std = torch.chunk(out, 2, dim=-1)
        std = torch.exp(std)

        pred_dist = Normal(mean, std)
        loss = - pred_dist.log_prob(batch.yt).sum(-1).mean()
        
        outs = AttrDict()
        outs.loss = loss
        return outs

    def predict(self, xc, yc, xt, num_samples=None):
        if xc.shape[-3] != xt.shape[-3]:
            xt = xt.transpose(-3, -2)

        batch = AttrDict()
        batch.xc = xc
        batch.yc = yc
        batch.xt = xt
        batch.yt = torch.zeros((xt.shape[0], xt.shape[1], yc.shape[2]), device='cuda')

        num_context = xc.shape[1]

        # in evaluation tnpa = tnpd because we only have 1 target point to predict
        out_encoder = self.encode(batch, autoreg=False)
        out = self.predictor(out_encoder)
        mean, std = torch.chunk(out, 2, dim=-1)
        std = torch.exp(std)
        
        return Normal(mean, std)