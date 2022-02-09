import torch
import torch.nn as nn
from torch.distributions import kl_divergence
from attrdict import AttrDict

from utils.misc import stack, logmeanexp
from models.modules import CrossAttnEncoder, PoolingEncoder, Decoder


class ANP(nn.Module):
    def __init__(self,
                 dim_x=1,
                 dim_y=1,
                 dim_hid=128,
                 dim_lat=128,
                 enc_v_depth=4,
                 enc_qk_depth=2,
                 enc_pre_depth=4,
                 enc_post_depth=2,
                 dec_depth=3):
        super(ANP, self).__init__()

        self.denc = CrossAttnEncoder(
            dim_x=dim_x,
            dim_y=dim_y,
            dim_hid=dim_hid,
            self_attn=True,
            v_depth=enc_v_depth,
            qk_depth=enc_qk_depth)

        self.lenc = PoolingEncoder(
            dim_x=dim_x,
            dim_y=dim_y,
            dim_hid=dim_hid,
            dim_lat=dim_lat,
            self_attn=True,
            pre_depth=enc_pre_depth,
            post_depth=enc_post_depth)

        self.dec = Decoder(
            dim_x=dim_x,
            dim_y=dim_y,
            dim_enc=dim_hid + dim_lat,
            dim_hid=dim_hid,
            depth=dec_depth)

    def predict(self, xc, yc, xt, z=None, num_samples=None):
        # botorch 사용하기 위해 추가된 statement
        if xc.shape[-3] != xt.shape[-3]:
            xt = xt.transpose(-3, -2)

        theta = stack(self.denc(xc, yc, xt), num_samples)
        if z is None:
            pz = self.lenc(xc, yc)
            z = pz.rsample() if num_samples is None \
                else pz.rsample([num_samples])
        z = stack(z, xt.shape[-2], dim=-2)
        encoded = torch.cat([theta, z], -1)
        return self.dec(encoded, stack(xt, num_samples))

    def forward(self, batch, num_samples=None, reduce_ll=True):
        outs = AttrDict()

        if self.training:
            pz = self.lenc(batch.xc, batch.yc)
            qz = self.lenc(batch.x, batch.y)
            z = qz.rsample() if num_samples is None else \
                qz.rsample([num_samples])
            py = self.predict(batch.xc, batch.yc, batch.x,
                              z=z, num_samples=num_samples)

            if num_samples > 1:
                # K * B * N
                recon = py.log_prob(stack(batch.y, num_samples)).sum(-1)
                # K * B
                log_qz = qz.log_prob(z).sum(-1)
                log_pz = pz.log_prob(z).sum(-1)

                # K * B
                log_w = recon.sum(-1) + log_pz - log_qz

                outs.loss = -logmeanexp(log_w).mean() / batch.x.shape[-2]
            else:
                outs.recon = py.log_prob(batch.y).sum(-1).mean()
                outs.kld = kl_divergence(qz, pz).sum(-1).mean()
                outs.loss = -outs.recon + outs.kld / batch.x.shape[-2]

        else:
            py = self.predict(batch.xc, batch.yc, batch.x, num_samples=num_samples)

            if num_samples is None:
                ll = py.log_prob(batch.y).sum(-1)
            else:
                y = torch.stack([batch.y] * num_samples)
                if reduce_ll:
                    ll = logmeanexp(py.log_prob(y).sum(-1))
                else:
                    ll = py.log_prob(y).sum(-1)

            num_ctx = batch.xc.shape[-2]

            if reduce_ll:
                outs.ctx_ll = ll[..., :num_ctx].mean()
                outs.tar_ll = ll[..., num_ctx:].mean()
            else:
                outs.ctx_ll = ll[..., :num_ctx]
                outs.tar_ll = ll[..., num_ctx:]

        return outs
