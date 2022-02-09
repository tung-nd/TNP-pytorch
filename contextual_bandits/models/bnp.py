import torch
import torch.nn as nn
from attrdict import AttrDict

from models.cnp import CNP
from utils.misc import stack, logmeanexp
from utils.sampling import sample_with_replacement as SWR, sample_subset


class BNP(CNP):
    def __init__(self, *args, **kwargs):
        super(BNP, self).__init__(*args, **kwargs)
        self.dec.add_ctx(2 * kwargs['dim_hid'])

    def encode(self, xc, yc, xt, mask=None):
        encoded = torch.cat([
            self.enc1(xc, yc, mask=mask),
            self.enc2(xc, yc, mask=mask)], -1)

        return stack(encoded, num_samples=xt.shape[-2], dim=-2)

    def predict(self, xc, yc, xt, num_samples=None, return_base=False, get_bootstrap=False):
        # botorch 사용하기 위해 추가된 statement
        if xc.shape[-3] != xt.shape[-3]:
            xt = xt.transpose(-3, -2)

        with torch.no_grad():
            bxc, byc = SWR(xc, yc, num_samples=num_samples)
            sxc, syc = stack(xc, num_samples=num_samples), stack(yc, num_samples=num_samples)

            encoded = self.encode(bxc, byc, sxc)
            py_res = self.dec(encoded, sxc)

            mu, sigma = py_res.mean, py_res.scale
            res = SWR((syc - mu)/sigma).detach()
            res = (res - res.mean(-2, keepdim=True))

            bxc = sxc
            byc = mu + sigma * res

        if get_bootstrap:
            return bxc, byc

        encoded_base = self.encode(xc, yc, xt)

        sxt = stack(xt, num_samples=num_samples)
        encoded_bs = self.encode(bxc, byc, sxt)

        py = self.dec(stack(encoded_base, num_samples),
                      sxt, ctx=encoded_bs)

        if self.training or return_base:
            py_base = self.dec(encoded_base, xt)
            return py_base, py
        else:
            return py

    def forward(self, batch, num_samples=None, reduce_ll=True):
        outs = AttrDict()

        def compute_ll(dist, y):
            loglikelihood = dist.log_prob(y).sum(-1)
            if loglikelihood.dim() == 3 and reduce_ll:
                loglikelihood = logmeanexp(loglikelihood)
            return loglikelihood

        if self.training:
            py_base, py = self.predict(batch.xc, batch.yc, batch.x,
                                       num_samples=num_samples)

            outs.ll_base = compute_ll(py_base, batch.y).mean()
            outs.ll = compute_ll(py, batch.y).mean()
            outs.loss = - outs.ll_base - outs.ll
        else:
            py = self.predict(batch.xc, batch.yc, batch.x,
                              num_samples=num_samples)
            ll = compute_ll(py, batch.y)
            num_ctx = batch.xc.shape[-2]
            if reduce_ll:
                outs.ctx_loss = ll[...,:num_ctx].mean()
                outs.tar_loss = ll[...,num_ctx:].mean()
            else:
                outs.ctx_loss = ll[...,:num_ctx]
                outs.tar_loss = ll[...,num_ctx:]

        return outs
