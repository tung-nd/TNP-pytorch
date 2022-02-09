import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from models.attention import MultiHeadAttn, SelfAttn


__all__ = ['PoolingEncoder', 'CrossAttnEncoder', 'Decoder', 'NeuBootsEncoder', 'NeuCrossAttnEncoder']


def build_mlp(dim_in, dim_hid, dim_out, depth):
    modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
    for _ in range(depth-2):
        modules.append(nn.Linear(dim_hid, dim_hid))
        modules.append(nn.ReLU(True))
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)


class PoolingEncoder(nn.Module):
    """
    (x, y) -> out, out can be a distribution or a single vector, [B,N,Eh] -> [B,Eh]
    dim_h: dimension of the hidden layer and also the output if dim_lat is None (no latent)
    dim_lat: dimension of latent, if None then this is deterministic encoder
    self_attn: if use self attention
    pre_depth: depth of (x_i, y_i) -> s_i
    post_depth: depth of s -> out
    """
    def __init__(self, dim_x=1, dim_y=1,
            dim_hid=128, dim_lat=None, self_attn=False,
            pre_depth=4, post_depth=2):
        super().__init__()

        self.use_lat = dim_lat is not None

        self.net_pre = build_mlp(dim_x+dim_y, dim_hid, dim_hid, pre_depth) \
                if not self_attn else \
                nn.Sequential(
                        build_mlp(dim_x+dim_y, dim_hid, dim_hid, pre_depth-2),
                        nn.ReLU(True),
                        SelfAttn(dim_hid, dim_hid))

        self.net_post = build_mlp(dim_hid, dim_hid,
                2*dim_lat if self.use_lat else dim_hid,
                post_depth)

    def forward(self, xc, yc, mask=None):
            out = self.net_pre(torch.cat([xc, yc], -1))  # [B,N,Eh]
            if mask is None:
                out = out.mean(-2)  # [B,Eh]
            else:
                mask = mask.to(xc.device)
                out = (out * mask.unsqueeze(-1)).sum(-2) / \
                        (mask.sum(-1, keepdim=True).detach() + 1e-5)
            if self.use_lat:
                mu, sigma = self.net_post(out).chunk(2, -1)
                sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
                return Normal(mu, sigma)
            else:
                return self.net_post(out)  # [B,Eh]


class CrossAttnEncoder(nn.Module):
    """
    (xc, yc, xt) -> out, out can be a distribution or a single vector, [B,N,Eh] -> [B,N,Eh]
    dim_h: dimension of the hidden layer and also the output if dim_lat is None (no latent)
    dim_lat: dimension of latent, if None then this is deterministic encoder
    self_attn: if use self attention
    v_depth: depth of (x_i, y_i) -> s_i for i in context
    qk_depth: depth of xt -> q and xc -> k
    """
    def __init__(self, dim_x=1, dim_y=1, dim_hid=128,
            dim_lat=None, self_attn=True,
            v_depth=4, qk_depth=2, neuboots=False):
        super().__init__()
        self.use_lat = dim_lat is not None

        self.neuboots = neuboots
        dim_v = dim_x + dim_y
        if neuboots:
            dim_v += 1  # w 차원 1 더해주

        if not self_attn:
            self.net_v = build_mlp(dim_v, dim_hid, dim_hid, v_depth)
        else:
            self.net_v = build_mlp(dim_v, dim_hid, dim_hid, v_depth-2)
            self.self_attn = SelfAttn(dim_hid, dim_hid)

        self.net_qk = build_mlp(dim_x, dim_hid, dim_hid, qk_depth)

        self.attn = MultiHeadAttn(dim_hid, dim_hid, dim_hid,
                2*dim_lat if self.use_lat else dim_hid)

    def forward(self, xc, yc, xt, w=None, mask=None):
        q, k = self.net_qk(xt), self.net_qk(xc)

        if self.neuboots:
            v = self.net_v(torch.cat([xc, yc, w], -1))
        else:
            v = self.net_v(torch.cat([xc, yc], -1))

        if hasattr(self, 'self_attn'):
            v = self.self_attn(v, mask=mask)
        if self.neuboots:
            v = v * w

        out = self.attn(q, k, v, mask=mask)
        if self.use_lat:
            mu, sigma = out.chunk(2, -1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return out


class NeuBootsEncoder(nn.Module):

    def __init__(self, dim_x=1, dim_y=1,
            dim_hid=128, dim_lat=None, self_attn=False,
            pre_depth=4, post_depth=2,
            yenc=True, wenc=True, wagg=True):
        super().__init__()

        self.use_lat = dim_lat is not None
        self.yenc = yenc
        self.wenc = wenc
        self.wagg = wagg
        dim_in = dim_x
        if yenc:
            dim_in += dim_y
        if wenc:
            dim_in += 1
        self.net_pre = build_mlp(dim_in, dim_hid, dim_hid, pre_depth) \
                if not self_attn else \
                nn.Sequential(
                        build_mlp(dim_in, dim_hid, dim_hid, pre_depth-2),
                        nn.ReLU(True),
                        SelfAttn(dim_hid, dim_hid))

        self.net_post = build_mlp(dim_hid, dim_hid,
                2*dim_lat if self.use_lat else dim_hid,
                post_depth)

    def forward(self, xc, yc=None, w=None):

        device = xc.device
        if not self.yenc:
            _yc = torch.tensor([]).to(device)
        else:
            _yc = yc
        if not self.wenc:
            _w = torch.tensor([]).to(device)
        else:
            _w = w

        # xc: [B,Nbs,N,Dx]
        # yc: [B,Nbs,N,Dy]
        # w: [B,Nbs,N,1]
        """
        Encoder
        """
        input = torch.cat([xc, _yc, _w], -1)  # [B,Nbs,N,?]
        output = self.net_pre(input)  # [B,Nbs,N,Eh]

        """
        Aggregation
        """
        if self.wagg:
            out = (output * w).mean(-2)  # [B,Nbs,Eh]
        else:
            out = output.mean(-2)  # [B,Nbs,Eh] : aggregation of context repr

        """
        Decoder
        """
        if self.use_lat:
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return self.net_post(out)  # [B,Eh]


class NeuCrossAttnEncoder(nn.Module):

    def __init__(self, dim_x=1, dim_y=1, dim_hid=128,
            dim_lat=None, self_attn=True,
            v_depth=4, qk_depth=2):
        super().__init__()
        self.use_lat = dim_lat is not None

        if not self_attn:
            self.net_v = build_mlp(dim_x + dim_y, dim_hid, dim_hid, v_depth)
        else:
            self.net_v = build_mlp(dim_x + dim_y, dim_hid, dim_hid, v_depth-2)
            self.self_attn = SelfAttn(dim_hid, dim_hid)

        self.net_qk = build_mlp(dim_x, dim_hid, dim_hid, qk_depth)

        self.attn = MultiHeadAttn(dim_hid, dim_hid, dim_hid,
                2*dim_lat if self.use_lat else dim_hid)

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


class Decoder(nn.Module):
    def __init__(self, dim_x=1, dim_y=1,
            dim_enc=128, dim_hid=128, depth=3, neuboots=False):
        super().__init__()
        self.fc = nn.Linear(dim_x+dim_enc, dim_hid)
        self.dim_hid = dim_hid
        self.neuboots = neuboots

        modules = [nn.ReLU(True)]
        for _ in range(depth-2):
            modules.append(nn.Linear(dim_hid, dim_hid))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dim_hid, dim_y if neuboots else 2*dim_y))
        self.mlp = nn.Sequential(*modules)

    def add_ctx(self, dim_ctx):
        self.dim_ctx = dim_ctx
        self.fc_ctx = nn.Linear(dim_ctx, self.dim_hid, bias=False)

    def forward(self, encoded, x, ctx=None):

        packed = torch.cat([encoded, x], -1)  # [B,(Nbs,)Nt,2Eh+Dx]
        hid = self.fc(packed)  # [B,(Nbs,)Nt,Dh]
        if ctx is not None:
            hid = hid + self.fc_ctx(ctx)  # [B,(Nbs,)Nt,Dh]
        out = self.mlp(hid)  # [B,(Nbs,)Nt,2Dy]
        if self.neuboots:
            return out  # [B,(Nbs,)Nt,2Dy]
        else:
            mu, sigma = out.chunk(2, -1)  # [B,Nt,Dy] each
            sigma = 0.1 + 0.9 * F.softplus(sigma)
            # sigma = F.softplus(sigma)
            return Normal(mu, sigma)  # Normal([B,Nt,Dy])