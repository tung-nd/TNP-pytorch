import torch
import torch.nn as nn

from models.modules import build_mlp


class TNP(nn.Module):
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
        bound_std
    ):
        super(TNP, self).__init__()

        self.embedder = build_mlp(dim_x + dim_y, d_model, d_model, emb_depth)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.bound_std = bound_std

    def construct_input(self, batch, autoreg=False):
        x_y_ctx = torch.cat((batch.xc, batch.yc), dim=-1)
        x_0_tar = torch.cat((batch.xt, torch.zeros_like(batch.yt)), dim=-1)
        if not autoreg:
            inp = torch.cat((x_y_ctx, x_0_tar), dim=1)
        else:
            if self.training and self.bound_std:
                yt_noise = batch.yt + 0.05 * torch.randn_like(batch.yt) # add noise to the past to smooth the model
                x_y_tar = torch.cat((batch.xt, yt_noise), dim=-1)
            else:
                x_y_tar = torch.cat((batch.xt, batch.yt), dim=-1)
            inp = torch.cat((x_y_ctx, x_y_tar, x_0_tar), dim=1)
        return inp

    def create_mask(self, batch, autoreg=False):
        num_ctx = batch.xc.shape[1]
        num_tar = batch.xt.shape[1]
        num_all = num_ctx + num_tar
        if not autoreg:
            mask = torch.zeros(num_all, num_all, device='cuda').fill_(float('-inf'))
            mask[:, :num_ctx] = 0.0
        else:
            mask = torch.zeros((num_all+num_tar, num_all+num_tar), device='cuda').fill_(float('-inf'))
            mask[:, :num_ctx] = 0.0 # all points attend to context points
            mask[num_ctx:num_all, num_ctx:num_all].triu_(diagonal=1) # each real target point attends to itself and precedding real target points
            mask[num_all:, num_ctx:num_all].triu_(diagonal=0) # each fake target point attends to preceeding real target points

        return mask, num_tar

    def construct_input_pretrain(self, batch):
        if self.training and self.bound_std:
            yt_noise = batch.yt + 0.05 * torch.randn_like(batch.yt) # add noise to the past to smooth the model
            x_y_tar = torch.cat((batch.xt, yt_noise), dim=-1)
        else:
            x_y_tar = torch.cat((batch.xt, batch.yt), dim=-1)

        x_0_tar = torch.cat((batch.xt, torch.zeros_like(batch.yt)), dim=-1)[:, 1:]
        inp = torch.cat((x_y_tar, x_0_tar), dim=1)
        return inp

    def create_mask_pretrain(self, batch):
        num_points = batch.xt.shape[1]

        mask = torch.zeros((2*num_points-1, 2*num_points-1), device='cuda').fill_(float('-inf'))
        mask[:num_points, :num_points].triu_(diagonal=1)
        mask[num_points:, 1:num_points].triu_(diagonal=0)
        mask[num_points:, 0] = 0.0

        return mask, num_points-1

    def encode(self, batch, autoreg=False, pretrain=False):
        if pretrain:
            inp = self.construct_input_pretrain(batch)
            mask, num_tar = self.create_mask_pretrain(batch)
        else:
            inp = self.construct_input(batch, autoreg)
            mask, num_tar = self.create_mask(batch, autoreg)
        embeddings = self.embedder(inp)
        out = self.encoder(embeddings, mask=mask)
        return out[:, -num_tar:]