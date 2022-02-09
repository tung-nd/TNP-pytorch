import torch

def gather(items, idxs):
    K = idxs.shape[0]
    idxs = idxs.to(items[0].device)
    gathered = []
    for item in items:
        gathered.append(torch.gather(
            torch.stack([item]*K), -2,
            torch.stack([idxs]*item.shape[-1], -1)).squeeze(0))
    return gathered[0] if len(gathered) == 1 else gathered

def sample_subset(*items, r_N=None, num_samples=None):
    r_N = r_N or torch.rand(1).item()
    K = num_samples or 1
    N = items[0].shape[-2]
    Ns = min(max(1, int(r_N * N)), N-1)
    batch_shape = items[0].shape[:-2]
    idxs = torch.rand((K,)+batch_shape+(N,)).argsort(-1)
    return gather(items, idxs[...,:Ns]), gather(items, idxs[...,Ns:])

def sample_with_replacement(*items, num_samples=None, r_N=1.0, N_s=None):
    K = num_samples or 1
    N = items[0].shape[-2]
    N_s = N_s or max(1, int(r_N * N))
    batch_shape = items[0].shape[:-2]
    idxs = torch.randint(N, size=(K,)+batch_shape+(N_s,))
    return gather(items, idxs)

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