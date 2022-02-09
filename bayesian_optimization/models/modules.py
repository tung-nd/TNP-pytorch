import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from models.attention import MultiHeadAttn, SelfAttn


def build_mlp(dim_in, dim_hid, dim_out, depth):
    modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(inplace=True)]
    for _ in range(depth - 2):
        modules.append(nn.Linear(dim_hid, dim_hid))
        modules.append(nn.ReLU(inplace=True))
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)


class PoolingEncoder(nn.Module):

    def __init__(self,
                 dim_x=1,
                 dim_y=1,
                 dim_hid=128,
                 dim_lat=None,
                 self_attn=False,
                 pre_depth=4,
                 post_depth=2):
        super(PoolingEncoder, self).__init__()

        self.use_lat = dim_lat is not None

        if not self_attn:
            self.net_pre = build_mlp(dim_x + dim_y, dim_hid, dim_hid, pre_depth)
        else:
            self.net_pre = nn.Sequential(
                build_mlp(dim_x + dim_y, dim_hid, dim_hid, pre_depth - 2),
                nn.ReLU(True),
                SelfAttn(dim_hid, dim_hid))

        self.net_post = build_mlp(dim_hid, dim_hid,
                                  2 * dim_lat if self.use_lat else dim_hid,
                                  post_depth)

    def forward(self, xc, yc, mask=None):
        out = self.net_pre(torch.cat([xc, yc], dim=-1))
        if mask is None:
            # aggregator
            out = out.mean(dim=-2)
        else:
            mask = mask.to(xc.device)
            out = (out * mask.unsqueeze(-1)).sum(-2) / \
                  (mask.sum(dim=-1, keepdim=True).detach() + 1e-5)

        if self.use_lat:
            mu, sigma = self.net_post(out).chunk(2, dim=-1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return self.net_post(out)


class CrossAttnEncoder(nn.Module):
    def __init__(self,
                 dim_x=1,
                 dim_y=1,
                 dim_hid=128,
                 dim_lat=None,
                 self_attn=True,
                 v_depth=4,
                 qk_depth=2):
        super(CrossAttnEncoder, self).__init__()

        self.use_lat = dim_lat is not None

        if not self_attn:
            self.net_v = build_mlp(dim_x + dim_y, dim_hid, dim_hid, v_depth)
        else:
            self.net_v = build_mlp(dim_x + dim_y, dim_hid, dim_hid, v_depth - 2)
            self.self_attn = SelfAttn(dim_hid, dim_hid)

        self.net_qk = build_mlp(dim_x, dim_hid, dim_hid, qk_depth)

        self.attn = MultiHeadAttn(dim_hid, dim_hid, dim_hid,
                                  2 * dim_lat if self.use_lat else dim_hid)

    def forward(self, xc, yc, xt, mask=None):
        q, k = self.net_qk(xt), self.net_qk(xc)
        v = self.net_v(torch.cat([xc, yc], -1))

        if hasattr(self, 'self_attn'):
            v = self.self_attn(v, mask=mask)

        out = self.attn(q, k, v, mask=mask)
        if self.use_lat:
            mu, sigma = out.chunk(2, -1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return out


class Decoder(nn.Module):
    def __init__(self,
                 dim_x=1,
                 dim_y=1,
                 dim_enc=128,
                 dim_hid=128,
                 depth=3,
                 neuboots=False):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(dim_x + dim_enc, dim_hid)
        self.dim_hid = dim_hid
        self.neuboots = neuboots

        modules = [nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            modules.append(nn.Linear(dim_hid, dim_hid))
            modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Linear(dim_hid, dim_y if neuboots else 2 * dim_y))
        self.mlp = nn.Sequential(*modules)

        self.dim_ctx = None
        self.fc_ctx = None

    # Adaptation layer
    def add_ctx(self, dim_ctx):
        self.dim_ctx = dim_ctx
        self.fc_ctx = nn.Linear(dim_ctx, self.dim_hid, bias=False)

    def forward(self, encoded, xt, ctx=None):
        packed = torch.cat([encoded, xt], dim=-1)
        hid = self.fc(packed)

        if ctx is not None:
            hid += self.fc_ctx(ctx)
        out = self.mlp(hid)

        if self.neuboots:
            return out
        else:
            mu, sigma = out.chunk(2, dim=-1)
            sigma = 0.1 + 0.9 * F.softplus(sigma)
            return Normal(mu, sigma)


class NeuCrossAttnEncoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128,
                 dim_lat=None, self_attn=True,
                 v_depth=4, qk_depth=2):
        super().__init__()
        self.use_lat = dim_lat is not None

        if not self_attn:
            self.net_v = build_mlp(dim_x + dim_y, dim_hid, dim_hid, v_depth)
        else:
            self.net_v = build_mlp(dim_x + dim_y, dim_hid, dim_hid, v_depth - 2)
            self.self_attn = SelfAttn(dim_hid, dim_hid)

        self.net_qk = build_mlp(dim_x, dim_hid, dim_hid, qk_depth)

        self.attn = MultiHeadAttn(dim_hid, dim_hid, dim_hid,
                                  2 * dim_lat if self.use_lat else dim_hid)

    def forward(self, xc, yc, xt, w, mask=None):
        q, k = self.net_qk(xt), self.net_qk(xc)
        v = self.net_v(torch.cat([xc, yc], -1))

        if hasattr(self, 'self_attn'):
            v = self.self_attn(v, mask=mask)

        v = v * w
        out = self.attn(q, k, v, mask=mask)
        if self.use_lat:
            mu, sigma = out.chunk(2, -1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return out


class NeuBootsEncoder(nn.Module):
    def __init__(self,
                 dim_x=1,
                 dim_y=1,
                 dim_hid=128,
                 dim_lat=None,
                 self_attn=False,
                 pre_depth=4,
                 post_depth=2,
                 yenc=False,
                 wenc=False,
                 wagg=None):
        super(NeuBootsEncoder, self).__init__()

        self.use_lat = dim_lat is not None
        self.yenc = yenc
        self.wenc = wenc
        self.wagg = wagg
        dim_in = dim_x

        if yenc:
            dim_in += dim_y
        if wenc:
            dim_in += 1

        if not self_attn:
            self.net_pre = build_mlp(dim_in, dim_hid, dim_hid, pre_depth)
        else:
            self.net_pre = nn.Sequential(build_mlp(dim_in, dim_hid, dim_hid, pre_depth - 2),
                                         nn.ReLU(True), SelfAttn(dim_hid, dim_hid))

        if 'l2a' in self.wagg:
            self.agg = nn.Linear(dim_hid, dim_hid)
            self.agg_activation = nn.ReLU()

        self.net_post = build_mlp(dim_hid, dim_hid,
                                  2 * dim_lat if self.use_lat else dim_hid,
                                  post_depth)

    def forward(self, xc, yc=None, w=None):
        if not self.yenc:
            _yc = torch.tensor([]).to(xc.device)
        else:
            _yc = yc

        if not self.wenc:
            _w = torch.tensor([]).to(xc.device)
        else:
            _w = w

        out = self.net_pre(torch.cat([xc, _yc, _w], dim=-1))
        # aggregation
        if self.wagg == 'mean':
            out = (out * w).mean(dim=-2)
        elif self.wagg == 'max':
            out = (out * w).max(dim=-2).values
        elif self.wagg == 'l2a':
            out = self.agg_activation(self.agg(out * w)).max(dim=-2).values
        elif self.wagg == 'l2a-mean':
            out = self.agg_activation(self.agg(out * w)).mean(dim=-2)
        else:
            out = out.mean(dim=-2)

        if self.use_lat:
            mu, sigma = self.net_post(out).chunk(2, dim=-1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return self.net_post(out)


if __name__ == "__main__":
    net = build_mlp(2, 128, 128, 4)
    print(list(net.parameters()))
