import torch
from attrdict import AttrDict
from torch.distributions import StudentT

def img_to_task(img, num_ctx=None,
        max_num_points=None, target_all=False, t_noise=None):

    B, C, H, W = img.shape
    num_pixels = H*W
    img = img.view(B, C, -1)

    if t_noise is not None:
        if t_noise == -1:
            t_noise = 0.09 * torch.rand(img.shape)
        img += t_noise * StudentT(2.1).rsample(img.shape)

    batch = AttrDict()
    max_num_points = max_num_points or num_pixels
    num_ctx = num_ctx or \
            torch.randint(low=3, high=max_num_points-3, size=[1]).item()
    num_tar = max_num_points - num_ctx if target_all else \
            torch.randint(low=3, high=max_num_points-num_ctx, size=[1]).item()
    num_points = num_ctx + num_tar
    idxs = torch.cuda.FloatTensor(B, num_pixels).uniform_().argsort(-1)[...,:num_points].to(img.device)
    x1, x2 = idxs//W, idxs%W
    batch.x = torch.stack([
        2*x1.float()/(H-1) - 1,
        2*x2.float()/(W-1) - 1], -1).to(img.device)
    batch.y = (torch.gather(img, -1, idxs.unsqueeze(-2).repeat(1, C, 1))\
            .transpose(-2, -1) - 0.5).to(img.device)

    batch.xc = batch.x[:,:num_ctx]
    batch.xt = batch.x[:,num_ctx:]
    batch.yc = batch.y[:,:num_ctx]
    batch.yt = batch.y[:,num_ctx:]

    return batch

def coord_to_img(x, y, shape):
    x = x.cpu()
    y = y.cpu()
    B = x.shape[0]
    C, H, W = shape

    I = torch.zeros(B, 3, H, W)
    I[:,0,:,:] = 0.61
    I[:,1,:,:] = 0.55
    I[:,2,:,:] = 0.71

    x1, x2 = x[...,0], x[...,1]
    x1 = ((x1+1)*(H-1)/2).round().long()
    x2 = ((x2+1)*(W-1)/2).round().long()
    for b in range(B):
        for c in range(3):
            I[b,c,x1[b],x2[b]] = y[b,:,min(c,C-1)]

    return I

def task_to_img(xc, yc, xt, yt, shape):
    xc = xc.cpu()
    yc = yc.cpu()
    xt = xt.cpu()
    yt = yt.cpu()

    B = xc.shape[0]
    C, H, W = shape

    xc1, xc2 = xc[...,0], xc[...,1]
    xc1 = ((xc1+1)*(H-1)/2).round().long()
    xc2 = ((xc2+1)*(W-1)/2).round().long()

    xt1, xt2 = xt[...,0], xt[...,1]
    xt1 = ((xt1+1)*(H-1)/2).round().long()
    xt2 = ((xt2+1)*(W-1)/2).round().long()

    task_img = torch.zeros(B, 3, H, W).to(xc.device)
    task_img[:,2,:,:] = 1.0
    task_img[:,1,:,:] = 0.4
    for b in range(B):
        for c in range(3):
            task_img[b,c,xc1[b],xc2[b]] = yc[b,:,min(c,C-1)] + 0.5
    task_img = task_img.clamp(0, 1)

    completed_img = task_img.clone()
    for b in range(B):
        for c in range(3):
            completed_img[b,c,xt1[b],xt2[b]] = yt[b,:,min(c,C-1)] + 0.5
    completed_img = completed_img.clamp(0, 1)

    return task_img, completed_img
