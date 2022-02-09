import torch

def gather(items, idxs, reduce=True):
    K = idxs.shape[0]  # Ns
    idxs = idxs.to(items[0].device)  # [Ns,B,N]
    gathered = []  # [Ns,B,N,D]
    for item in items:  # [B,N,D]
        _gathered = torch.gather(
            torch.stack([item] * K), -2,  # [Ns,B,N,D]
            torch.stack([idxs] * item.shape[-1], -1))
        gathered.append(_gathered.squeeze(0) if reduce else _gathered)  # [Ns,B,N,D]
    return gathered[0] if len(gathered) == 1 else gathered

def sample_subset(*items, r_N=None, num_samples=None):
    r_N = r_N or torch.rand(1).item()
    K = num_samples or 1
    N = items[0].shape[-2]
    Ns = min(max(1, int(r_N * N)), N-1)
    batch_shape = items[0].shape[:-2]
    idxs = torch.rand((K,)+batch_shape+(N,)).argsort(-1)
    return gather(items, idxs[...,:Ns]), gather(items, idxs[...,Ns:])

def sample_with_replacement(*items, num_samples=None, r_N=1.0, N_s=None, reduce=True):
    K = num_samples or 1  # Ns
    N = items[0].shape[-2]  # N
    N_s = N_s or max(1, int(r_N * N))  # N
    batch_shape = items[0].shape[:-2]  # B
    idxs = torch.randint(N, size=(K,)+batch_shape+(N_s,))  # [Ns,B,N]
    return gather(items, idxs, reduce)  # items: [B,N,D], idxs: [Ns,B,N]

def sample_mask(B, N, num_samples=None, min_num=3, prob=0.5):
    min_num = min(min_num, N)
    K = num_samples or 1
    fixed = torch.ones(K, B, min_num)
    if N - min_num > 0:
        rand = torch.bernoulli(prob*torch.ones(K, B, N-min_num))
        mask = torch.cat([fixed, rand], -1)
        return mask.squeeze(0)
    else:
        return fixed.squeeze(0)