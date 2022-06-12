import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from attrdict import AttrDict

from models.modules import build_mlp
from models.tnp import TNP


class TNPND(TNP):
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
        num_std_layers,
        cov_approx='cholesky',
        prj_dim=5,
        prj_depth=4,
        diag_depth=4
    ):
        super(TNPND, self).__init__(
            dim_x,
            dim_y,
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            dropout,
            num_layers,
        )

        assert cov_approx in ['cholesky', 'lowrank']
        self.cov_approx = cov_approx
        
        self.mean_net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y)
        )

        std_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.std_encoder = nn.TransformerEncoder(std_encoder_layer, num_std_layers)

        self.projector = build_mlp(d_model, dim_feedforward, prj_dim*dim_y, prj_depth)

        if cov_approx == 'lowrank':
            self.diag_net = build_mlp(d_model, dim_feedforward, dim_y, diag_depth)

    def decode(self, out_encoder, batch_size, dim_y, num_target):
        mean_target = self.mean_net(out_encoder).view(batch_size, -1)

        out_std_encoder = self.std_encoder(out_encoder)
        std_prj = self.projector(out_std_encoder)
        std_prj = std_prj.view((batch_size, num_target*dim_y, -1))
        if self.cov_approx == 'cholesky':
            std_tril = torch.bmm(std_prj, std_prj.transpose(1,2))
            std_tril = std_tril.tril()
            if self.emnist:
                diag_ids = torch.arange(num_target*dim_y, device='cuda')
                std_tril[:, diag_ids, diag_ids] = 0.05 + 0.95*torch.tanh(std_tril[:, diag_ids, diag_ids])
            pred_tar = torch.distributions.multivariate_normal.MultivariateNormal(mean_target, scale_tril=std_tril)
        else:
            diagonal = torch.exp(self.diag_net(out_encoder)).view((batch_size, -1, 1))
            std = torch.bmm(std_prj, std_prj.transpose(1,2)) + torch.diag_embed(diagonal.squeeze(-1))
            pred_tar = torch.distributions.multivariate_normal.MultivariateNormal(mean_target, covariance_matrix=std)

        return pred_tar

    def forward(self, batch, reduce_ll=True):
        batch_size = batch.x.shape[0]
        dim_y = batch.y.shape[-1]
        num_target = batch.xt.shape[1]

        out_encoder = self.encode(batch, autoreg=False)
        pred_tar = self.decode(out_encoder, batch_size, dim_y, num_target)

        outs = AttrDict()
        yt = batch.yt.reshape(batch.yt.shape[0], -1)
        outs.loss = - (pred_tar.log_prob(yt).mean() / num_target)
        return outs


    def predict(self, xc, yc, xt, num_samples=None):
        if xc.shape[-3] != xt.shape[-3]:
            xt = xt.transpose(-3, -2)

        batch = AttrDict()
        batch.xc = xc
        batch.yc = yc
        batch.xt = xt
        batch.yt = torch.zeros((xt.shape[0], xt.shape[1], yc.shape[2]), device='cuda')

        batch_size = xc.shape[0]
        dim_y = yc.shape[-1]
        num_target = batch.xt.shape[1]

        out_encoder = self.encode(batch, autoreg=False)
        pred_tar = self.decode(out_encoder, batch_size, dim_y, num_target)

        return Normal(
            pred_tar.mean.view(batch_size, num_target, -1),
            torch.diagonal(pred_tar.covariance_matrix, dim1=-2, dim2=-1).reshape(batch_size, num_target, -1)
        )