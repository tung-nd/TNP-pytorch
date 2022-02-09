import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttn(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim_out, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.fc_q = nn.Linear(dim_q, dim_out, bias=False)
        self.fc_k = nn.Linear(dim_k, dim_out, bias=False)
        self.fc_v = nn.Linear(dim_v, dim_out, bias=False)
        self.fc_out = nn.Linear(dim_out, dim_out)
        self.ln1 = nn.LayerNorm(dim_out)
        self.ln2 = nn.LayerNorm(dim_out)

    def scatter(self, x):
        return torch.cat(x.chunk(self.num_heads, -1), -3)

    def gather(self, x):
        return torch.cat(x.chunk(self.num_heads, -3), -1)

    def attend(self, q, k, v, mask=None):
        q_, k_, v_ = [self.scatter(x) for x in [q, k, v]]
        A_logits = q_ @ k_.transpose(-2, -1) / math.sqrt(self.dim_out)
        if mask is not None:
            mask = mask.bool().to(q.device)
            mask = torch.stack([mask]*q.shape[-2], -2)
            mask = torch.cat([mask]*self.num_heads, -3)
            A = torch.softmax(A_logits.masked_fill(mask, -float('inf')), -1)
            A = A.masked_fill(torch.isnan(A), 0.0)
        else:
            A = torch.softmax(A_logits, -1)
        return self.gather(A @ v_)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        out = self.ln1(q + self.attend(q, k, v, mask=mask))
        out = self.ln2(out + F.relu(self.fc_out(out)))
        return out

class SelfAttn(MultiHeadAttn):
    def __init__(self, dim_in, dim_out, num_heads=8):
        super().__init__(dim_in, dim_in, dim_in, dim_out, num_heads)

    def forward(self, x, mask=None):
        return super().forward(x, x, x, mask=mask)
