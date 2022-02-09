import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from models.attention import MultiHeadAttn, SelfAttn


__all__ = ['PoolingEncoder', 'CrossAttnEncoder', 'Decoder']


def build_mlp(dim_in, dim_hid, dim_out, depth):
    modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
    for _ in range(depth-2):
        modules.append(nn.Linear(dim_hid, dim_hid))
        modules.append(nn.ReLU(True))
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)


class PoolingEncoder(nn.Module):

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

    def __init__(self, dim_x=1, dim_y=1, dim_hid=128,
            dim_lat=None, self_attn=True,
            v_depth=4, qk_depth=2):
        super().__init__()
        self.use_lat = dim_lat is not None

        if not self_attn:
            self.net_v = build_mlp(dim_x+dim_y, dim_hid, dim_hid, v_depth)
        else:
            self.net_v = build_mlp(dim_x+dim_y, dim_hid, dim_hid, v_depth-2)
            self.self_attn = SelfAttn(dim_hid, dim_hid)

        self.net_qk = build_mlp(dim_x, dim_hid, dim_hid, qk_depth)

        self.attn = MultiHeadAttn(dim_hid, dim_hid, dim_hid,
                2*dim_lat if self.use_lat else dim_hid)

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

class NeuCrossAttnEncoder(nn.Module):

    def __init__(self, dim_x=1, dim_y=1, dim_hid=128,
            dim_lat=None, self_attn=True,
            v_depth=4, qk_depth=2):
        super().__init__()
        self.use_lat = dim_lat is not None

        if not self_attn:
            self.net_v = build_mlp(dim_x+dim_y, dim_hid, dim_hid, v_depth)
        else:
            self.net_v = build_mlp(dim_x+dim_y, dim_hid, dim_hid, v_depth-2)
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
            return Normal(mu, sigma)  # Normal([B,Nt,Dy])


class NeuBootsEncoder(nn.Module):

    def   __init__(self, dim_x=1, dim_y=1,
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

        if self.wagg == 'l2a':
            self.agg = nn.Linear(dim_hid,dim_hid)
            self.agg_activation = nn.ReLU()

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
        if self.wagg == 'mean':
            out = (output * w).mean(-2)  # [B,Nbs,Eh]
        elif self.wagg == 'max':
            out = (output * w).max(-2).values
        elif self.wagg == 'l2a':
            out = self.agg_activation(self.agg(output * w)).max(dim=-2).values
        else:
            out = output.mean(-2)   # --wagg None
            # [B,Nbs,Eh] : aggregation of context repr

        """
        Decoder
        """
        if self.use_lat:
            mu, sigma = self.net_post(out).chunk(2, -1)
            sigma = 0.1 + 0.9 * torch.sigmoid(sigma)
            return Normal(mu, sigma)
        else:
            return self.net_post(out)  # [B,Eh]


class CouplingLayer(nn.Module):
  """
  Implementation of the affine coupling layer in RealNVP
  paper.
  """

  def __init__(self, d_inp, d_model, nhead, dim_feedforward, orientation, num_layers):
    super().__init__()

    self.orientation = orientation

    self.embedder = build_mlp(d_inp, d_model, d_model, 2)
    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.0, batch_first=True)
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    self.ffn = build_mlp(d_model, dim_feedforward, d_inp*2, 2)

    self.scale_net = build_mlp(d_model, dim_feedforward, d_inp, 2)

  def coupling(self, x):
    embeddings = self.embedder(x)
    out_encoder = self.encoder(embeddings)
    s_t = self.ffn(out_encoder)
    scale = torch.sigmoid(self.scale_net(out_encoder))
    return s_t, scale

  def forward(self, x, logdet, invert=False):
    if not invert:
      x1, x2, mask = self.split(x)
      out, scale = self.coupling(x1)
      t, log_s = torch.chunk(out, 2, dim=-1)
      log_s = torch.tanh(log_s) / scale
      s = torch.exp(log_s)
      logdet += torch.sum(log_s.view(s.shape[0], -1), dim=-1)
      y1, y2 = x1, s * (x2 + t)
      return self.merge(y1, y2, mask), logdet

    # Inverse affine coupling layer
    y1, y2, mask = self.split(x)
    out, scale = self.coupling(y1)
    t, log_s = torch.chunk(out, 2, dim=-1)
    log_s = torch.tanh(log_s) / scale
    s = torch.exp(log_s)
    logdet -= torch.sum(log_s.view(s.shape[0], -1), dim=-1)
    x1, x2 = y1, y2 / s - t
    return self.merge(x1, x2, mask), logdet

  def split(self, x):
    assert x.shape[1] % 2 == 0
    mask = torch.zeros(x.shape[1], device='cuda')
    mask[::2] = 1.
    if self.orientation:
      mask = 1. - mask     # flip mask orientation

    x1, x2 = x[:, mask.bool()], x[:, (1-mask).bool()]
    return x1, x2, mask

  def merge(self, x1, x2, mask):
    x = torch.zeros((x2.shape[0], x1.shape[1]*2, x1.shape[2]), device='cuda')
    x[:, mask.bool()] = x1
    x[:, (1-mask).bool()] = x2
    return x

class NICE(nn.Module):
  def __init__(self, d_inp, d_model, nhead, dim_feedforward, num_layers_coupling=2, num_coupling_layers=2):
    super().__init__()

    # alternating mask orientations for consecutive coupling layers
    mask_orientations = [(i % 2 == 0) for i in range(num_coupling_layers)]

    self.coupling_layers = nn.ModuleList([
        CouplingLayer(
            d_inp, d_model, nhead, dim_feedforward, mask_orientations[i], num_layers_coupling
        ) for i in range(num_coupling_layers)
    ])


  def forward(self, x, invert=False):
    if not invert:
      z, log_det_jacobian = self.f(x)
      return z, log_det_jacobian

    return self.f_inverse(x)

  def f(self, x):
    z = x
    log_det_jacobian = 0
    for i, coupling_layer in enumerate(self.coupling_layers):
      z, log_det_jacobian = coupling_layer(z, log_det_jacobian)
    return z, log_det_jacobian

  def f_inverse(self, z):
    x = z
    for i, coupling_layer in reversed(list(enumerate(self.coupling_layers))):
      x, _ = coupling_layer(x, 0, invert=True)
    return x

# nice = NICE(1, 10, 1, 20, 2, 4).cuda()
# y = torch.randn((2, 4, 1), device='cuda')
# z, logdet = nice(y)
# y_prime = nice(z, True)
# print (y)
# print (z)
# print (y_prime)