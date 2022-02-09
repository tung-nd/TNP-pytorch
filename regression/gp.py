import os
import os.path as osp

import argparse
import yaml

import torch
import torch.nn as nn
import numpy as np

import math
import time
import matplotlib.pyplot as plt
from attrdict import AttrDict
from tqdm import tqdm
from copy import deepcopy

import uncertainty_toolbox as uct

from data.gp import *

from utils.misc import load_module, logmeanexp, stack
from utils.paths import results_path, datasets_path, evalsets_path
from utils.log import get_logger, RunningAverage

def main():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--mode', default='train')
    parser.add_argument('--expid', type=str, default='default')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)  # default(-1): device="cpu"

    # Data
    parser.add_argument('--max_num_points', type=int, default=50)

    # Model
    parser.add_argument('--model', type=str, default="tnpd")

    # Train
    parser.add_argument('--train_seed', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--train_num_samples', type=int, default=4)
    parser.add_argument('--train_num_bs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=1000)

    # Eval
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--eval_num_batches', type=int, default=3000)
    parser.add_argument('--eval_num_bs', type=int, default=50)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_samples', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)

    # Plot
    parser.add_argument('--plot_seed', type=int, default=0)
    parser.add_argument('--plot_batch_size', type=int, default=16)
    parser.add_argument('--plot_num_samples', type=int, default=30)
    parser.add_argument('--plot_num_bs', type=int, default=50)
    parser.add_argument('--plot_num_ctx', type=int, default=30)
    parser.add_argument('--plot_num_tar', type=int, default=10)
    parser.add_argument('--start_time', type=str, default=None)

    # OOD settings
    parser.add_argument('--eval_kernel', type=str, default='rbf')
    parser.add_argument('--t_noise', type=float, default=None)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.expid is not None:
        args.root = osp.join(results_path, 'gp', args.model, args.expid)
    else:
        args.root = osp.join(results_path, 'gp', args.model)

    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    with open(f'configs/gp/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp", "tnpd", "tnpa", "tnpnd"]:
        model = model_cls(**config)
    model.cuda()

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        eval(args, model)
    elif args.mode == 'eval_all_metrics':
        eval_all_metrics(args, model)
    elif args.mode == 'eval_multiple_runs':
        eval_multiple_runs(args, model)
    elif args.mode == 'eval_all_metrics_multiple_runs':
        eval_all_metrics_multiple_runs(args, model)
    elif args.mode == 'plot':
        plot(args, model)

def train(args, model):
    if osp.exists(args.root + '/ckpt.tar'):
        if args.resume is None:
            raise FileExistsError(args.root)
    else:
        os.makedirs(args.root, exist_ok=True)

    with open(osp.join(args.root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)

    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed(args.train_seed)

    sampler = GPSampler(RBFKernel())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_steps)

    if args.resume:
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_step = ckpt.step
    else:
        logfilename = os.path.join(args.root,
                f'train_{time.strftime("%Y%m%d-%H%M")}.log')
        start_step = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info(f"Experiment: {args.model}-{args.expid}")
        logger.info(f"Device: {args.gpu}\n")
        logger.info(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}\n')

    for step in range(start_step, args.num_steps+1):
        model.train()
        optimizer.zero_grad()
        batch = sampler.sample(
            batch_size=args.train_batch_size,
            max_num_points=args.max_num_points,
            device='cuda')
        
        if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp"]:
            outs = model(batch, num_samples=args.train_num_samples)
        else:
            outs = model(batch)

        outs.loss.backward()
        optimizer.step()
        scheduler.step()

        for key, val in outs.items():
            ravg.update(key, val)

        if step % args.print_freq == 0:
            line = f'{args.model}:{args.expid} step {step} '
            line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
            line += f"[train_loss] "
            line += ravg.info()
            logger.info(line)

            if step % args.eval_freq == 0:
                line = eval(args, model)
                logger.info(line + '\n')

            ravg.reset()

        if step % args.save_freq == 0 or step == args.num_steps:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.step = step + 1
            torch.save(ckpt, os.path.join(args.root, 'ckpt.tar'))

    args.mode = 'eval'
    eval(args, model)

def get_eval_path(args):
    path = osp.join(evalsets_path, 'gp')
    filename = f'{args.eval_kernel}-seed{args.eval_seed}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'
    return path, filename

def gen_evalset(args):
    if args.eval_kernel == 'rbf':
        kernel = RBFKernel()
    elif args.eval_kernel == 'matern':
        kernel = Matern52Kernel()
    elif args.eval_kernel == 'periodic':
        kernel = PeriodicKernel()
    else:
        raise ValueError(f'Invalid kernel {args.eval_kernel}')
    print(f"Generating Evaluation Sets with {args.eval_kernel} kernel")

    sampler = GPSampler(kernel, t_noise=args.t_noise, seed=args.eval_seed)
    batches = []
    for i in tqdm(range(args.eval_num_batches), ascii=True):
        batches.append(sampler.sample(
            batch_size=args.eval_batch_size,
            max_num_points=args.max_num_points,
            device='cuda'))

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path, filename = get_eval_path(args)
    if not osp.isdir(path):
        os.makedirs(path)
    torch.save(batches, osp.join(path, filename))

def eval(args, model):
    if args.mode == 'eval':
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'), map_location='cuda')
        model.load_state_dict(ckpt.model)
        if args.eval_logfile is None:
            eval_logfile = f'eval_{args.eval_kernel}'
            if args.t_noise is not None:
                eval_logfile += f'_tn_{args.t_noise}'
            eval_logfile += '.log'
        else:
            eval_logfile = args.eval_logfile
        filename = os.path.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)
    eval_batches = torch.load(osp.join(path, filename))

    if args.mode == "eval":
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed(args.eval_seed)

    ravg = RunningAverage()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_batches, ascii=True):
            for key, val in batch.items():
                batch[key] = val.cuda()
            if args.model in ["np", "anp", "bnp", "banp"]:
                outs = model(batch, args.eval_num_samples)
            else:
                outs = model(batch)

            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f'{args.model}:{args.expid} {args.eval_kernel} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    line += ravg.info()

    if logger is not None:
        logger.info(line)

    return line

def eval_multiple_runs(args, model):
    num_runs = 5

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)

    eval_batches = torch.load(osp.join(path, filename))

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    ravgs = [RunningAverage() for _ in range(num_runs)]
    with torch.no_grad():
        for i in range(num_runs):
            model_ = deepcopy(model)
            ckpt = torch.load(osp.join(results_path, 'gp', args.model, f'run{i+1}', 'ckpt.tar'), map_location='cuda')
            model_.load_state_dict(ckpt['model'])
            model_.eval()
            print ('Evaluating run %d' % (i+1))
            for batch in tqdm(eval_batches, ascii=True):
                for key, val in batch.items():
                    batch[key] = val.cuda()
                
                if args.model in ["np", "anp", "bnp", "banp"]:
                    outs = model_(batch, args.eval_num_samples)
                else:
                    outs = model_(batch)

                for key, val in outs.items():
                    ravgs[i].update(key, val)
        
        tar_ll_all = [ragv.sum['tar_ll'] / ragv.cnt['tar_ll'] for ragv in ravgs]
    
    line = f'{args.model}: {args.eval_kernel} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    line += f'\n'
    for i in range(num_runs):
        line += f'run {i+1}: tar_ll {tar_ll_all[i]}'
        line += '\n'
    line += f'average: {np.mean(tar_ll_all)}\n'
    line += f'std: {np.std(tar_ll_all)}'

    filename = f'eval_{args.eval_kernel}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += f'_{num_runs}runs'
    filename += '.log'
    logger = get_logger(osp.join(results_path, 'gp', args.model, filename), mode='w')

    logger.info(line)

def eval_all_metrics(args, model):
    ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'), map_location='cuda')
    model.load_state_dict(ckpt.model)
    if args.eval_logfile is None:
        eval_logfile = f'eval_{args.eval_kernel}'
        if args.t_noise is not None:
            eval_logfile += f'_tn_{args.t_noise}'
        eval_logfile += f'_all_metrics'
        eval_logfile += '.log'
    else:
        eval_logfile = args.eval_logfile
    filename = os.path.join(args.root, eval_logfile)
    logger = get_logger(filename, mode='w')

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)
    eval_batches = torch.load(osp.join(path, filename))

    if args.mode == "eval_all_metrics":
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed(args.eval_seed)

    model.eval()
    with torch.no_grad():
        ravgs = [RunningAverage() for _ in range(4)] # 4 types of metrics
        for batch in tqdm(eval_batches, ascii=True):
            for key, val in batch.items():
                batch[key] = val.cuda()
            if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp"]:
                outs = model.predict(batch.xc, batch.yc, batch.xt, num_samples=args.eval_num_samples)
                ll = model(batch, num_samples=args.eval_num_samples)
            elif args.model in ["tnpa", "tnpnd"]:
                outs = model.predict(
                    batch.xc, batch.yc, batch.xt,
                    num_samples=args.eval_num_samples
                )
                ll = model(batch)
            else:
                outs = model.predict(batch.xc, batch.yc, batch.xt)
                ll = model(batch)

            mean, std = outs.loc, outs.scale

            # shape: (num_samples, 1, num_points, 1)
            if mean.dim() == 4:
                # variance of samples (Law of Total Variance) - var(X) = E[var(X|Y)] + var(E[X|Y])
                # E[var(X|Y)] : average variability within each samples
                # var(E[X|Y]) : variability between samples
                var = std.pow(2).mean(dim=0) + mean.pow(2).mean(dim=0) - mean.mean(dim=0).pow(2)
                std = var.sqrt().squeeze(0)
                # mean of samples (Law of Total Expectations) - E[E[X|Y]] = E[X]
                mean = mean.mean(dim=0).squeeze(0)
            
            mean, std = mean.squeeze().cpu().numpy().flatten(), std.squeeze().cpu().numpy().flatten()
            yt = batch.yt.squeeze().cpu().numpy().flatten()

            acc = uct.metrics.get_all_accuracy_metrics(mean, yt, verbose=False)
            calibration = uct.metrics.get_all_average_calibration(mean, std, yt, num_bins=100, verbose=False)
            sharpness = uct.metrics.get_all_sharpness_metrics(std, verbose=False)
            scoring_rule = {'tar_ll': ll.tar_ll.item()}

            batch_metrics = [acc, calibration, sharpness, scoring_rule]
            for i in range(len(batch_metrics)):
                ravg, batch_metric = ravgs[i], batch_metrics[i]
                for k in batch_metric.keys():
                    ravg.update(k, batch_metric[k])

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f'{args.model}:{args.expid} {args.eval_kernel} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    
    line += '\n'

    for ravg in ravgs:
        line += ravg.info()
        line += '\n'

    if logger is not None:
        logger.info(line)

    return line

def eval_all_metrics_multiple_runs(args, model):
    num_runs = 5

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)

    eval_batches = torch.load(osp.join(path, filename))

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    all_metrics = [{}, {}, {}, {}] # acc, calibration, sharpness, scoring_rule

    with torch.no_grad():
        for i in range(num_runs):
            model_ = deepcopy(model)
            ckpt = torch.load(osp.join(results_path, 'gp', args.model, f'run{i+1}', 'ckpt.tar'), map_location='cuda')
            model_.load_state_dict(ckpt['model'])
            model_.eval()

            print ('Evaluating run %d' % (i+1))

            ragvs = [RunningAverage() for _ in range(4)] # 4 types of metrics

            for batch in tqdm(eval_batches, ascii=True):
                for key, val in batch.items():
                    batch[key] = val.cuda()

                if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp"]:
                    outs = model_.predict(batch.xc, batch.yc, batch.xt, num_samples=args.eval_num_samples)
                    ll = model_(batch, num_samples=args.eval_num_samples)
                elif args.model in ["tnpa", "tnpnd"]:
                    outs = model_.predict(
                        batch.xc, batch.yc, batch.xt,
                        num_samples=args.eval_num_samples
                    )
                    ll = model_(batch)
                else:
                    outs = model_.predict(batch.xc, batch.yc, batch.xt)
                    ll = model_(batch)

                mean, std = outs.loc, outs.scale

                # shape: (num_samples, 1, num_points, 1)
                if mean.dim() == 4:
                    var = std.pow(2).mean(dim=0) + mean.pow(2).mean(dim=0) - mean.mean(dim=0).pow(2)
                    std = var.sqrt().squeeze(0)
                    mean = mean.mean(dim=0).squeeze(0)
                
                mean, std = mean.squeeze().cpu().numpy().flatten(), std.squeeze().cpu().numpy().flatten()
                yt = batch.yt.squeeze().cpu().numpy().flatten()

                acc = uct.metrics.get_all_accuracy_metrics(mean, yt, verbose=False)
                calibration = uct.metrics.get_all_average_calibration(mean, std, yt, num_bins=100, verbose=False)
                sharpness = uct.metrics.get_all_sharpness_metrics(std, verbose=False)
                scoring_rule = {'tar_ll': ll.tar_ll.item()}

                batch_metrics = [acc, calibration, sharpness, scoring_rule]
                for i in range(len(batch_metrics)):
                    ragv, batch_metric = ragvs[i], batch_metrics[i]
                    for k in batch_metric.keys():
                        ragv.update(k, batch_metric[k])


            for i, ragv in enumerate(ragvs):
                for sub_metric in ragv.keys():
                    if sub_metric not in all_metrics[i].keys():
                        all_metrics[i][sub_metric] = [ragv.get(sub_metric)]
                    else:
                        all_metrics[i][sub_metric].append(ragv.get(sub_metric))
            

    line = f'{args.model}: {args.eval_kernel} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    line += f'\n'
    line += f'all metrics of {num_runs} runs:\n'

    for metric in all_metrics:
        line += str(metric)
        line += f'\n'

    for metric in all_metrics:
        for sub_metric in metric.keys():
            mean = np.mean(metric[sub_metric])
            std = np.std(metric[sub_metric])
            metric[sub_metric] = '%f +- %f' % (mean, std)
    
    line += f'mean and std of all metrics of {num_runs} runs:\n'

    for metric in all_metrics:
        line += str(metric)
        line += f'\n'

    filename = f'eval_{args.eval_kernel}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += f'_all_metrics_{num_runs}runs'
    filename += '.log'
    logger = get_logger(osp.join(results_path, 'gp', args.model, filename), mode='w')

    logger.info(line)

def plot(args, model):
    seed = args.plot_seed
    device_tmp = torch.device("cpu")
    num_bs = args.plot_num_bs
    num_smp = args.plot_num_samples
    loss=args.loss
    alpha = args.alpha
    beta = args.beta
    eps = args.eps

    if args.mode == "plot":
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'), map_location='cuda')
        model.load_state_dict(ckpt.model)
    model = model.cuda()

    def tnp(x):
        return x.squeeze().cpu().data.numpy()

    kernel = RBFKernel()
    sampler = GPSampler(kernel, t_noise=args.t_noise, seed=args.eval_seed)

    xp = torch.linspace(-2, 2, 200).cuda()
    batch = sampler.sample(
        batch_size=args.plot_batch_size,
        num_ctx=args.plot_num_ctx,
        num_tar=args.plot_num_tar,
        device='cuda',
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    Nc = batch.xc.size(1)
    Nt = batch.xt.size(1)

    model.eval()
    with torch.no_grad():
        if args.model in ["np", "anp", "bnp", "banp"]:
            outs = model(batch, num_smp, reduce_ll=False)
        else:
            outs = model(batch, reduce_ll=False)
        tar_loss = outs.tar_ll  # [Ns,B,Nt] ([B,Nt] for CNP)
        if args.model in ["cnp", "canp", "tnpd", "tnpa", "tnpnd"]:
            tar_loss = tar_loss.unsqueeze(0)  # [1,B,Nt]

        xt = xp[None, :, None].repeat(args.plot_batch_size, 1, 1)
        if args.model in ["np", "anp", "bnp", "banp", "tnpa", "tnpnd"]:
            pred = model.predict(batch.xc, batch.yc, xt, num_samples=num_smp)
        else:
            pred = model.predict(batch.xc, batch.yc, xt)
        
        mu, sigma = pred.mean, pred.scale

    if args.plot_batch_size > 1:
        nrows = max(args.plot_batch_size//4, 1)
        ncols = min(4, args.plot_batch_size)
        fig, axes = plt.subplots(nrows, ncols,
                figsize=(5*ncols, 5*nrows))
        axes = axes.flatten()
    else:
        fig = plt.figure(figsize=(5, 5))
        axes = [plt.gca()]

    # multi sample
    if mu.dim() == 4:
        for i, ax in enumerate(axes):
            for s in range(mu.shape[0]):
                ax.plot(tnp(xp), tnp(mu[s][i]), color='steelblue',
                        alpha=max(0.5/args.plot_num_samples, 0.1))
                ax.fill_between(tnp(xp), tnp(mu[s][i])-tnp(sigma[s][i]),
                        tnp(mu[s][i])+tnp(sigma[s][i]),
                        color='skyblue',
                        alpha=max(0.2/args.plot_num_samples, 0.02),
                        linewidth=0.0)
            ax.scatter(tnp(batch.xc[i]), tnp(batch.yc[i]),
                       color='k', label=f'context {Nc}', zorder=mu.shape[0] + 1)
            ax.scatter(tnp(batch.xt[i]), tnp(batch.yt[i]),
                       color='orchid', label=f'target {Nt}',
                       zorder=mu.shape[0] + 1)
            ax.legend()
            ax.set_title(f"tar_loss: {tar_loss[:, i, :].mean(): 0.4f}")
    else:
        for i, ax in enumerate(axes):
            ax.plot(tnp(xp), tnp(mu[i]), color='steelblue', alpha=0.5)
            ax.fill_between(tnp(xp), tnp(mu[i]-sigma[i]), tnp(mu[i]+sigma[i]),
                    color='skyblue', alpha=0.2, linewidth=0.0)
            ax.scatter(tnp(batch.xc[i]), tnp(batch.yc[i]),
                       color='k', label=f'context {Nc}')
            ax.scatter(tnp(batch.xt[i]), tnp(batch.yt[i]),
                       color='orchid', label=f'target {Nt}')
            ax.legend()
            ax.set_title(f"tar_loss: {tar_loss[:, i, :].mean(): 0.4f}")

    plt.suptitle(f"{args.expid}", y=0.995)
    plt.tight_layout()

    save_dir_1 = osp.join(args.root, f"plot_num{num_smp}-c{Nc}-t{Nt}-seed{seed}-{args.start_time}.jpg")
    file_name = "-".join([args.model, args.expid, f"plot_num{num_smp}",
                          f"c{Nc}", f"t{Nt}", f"seed{seed}", f"{args.start_time}.jpg"])
    if args.expid is not None:
        save_dir_2 = osp.join(results_path, "gp", "plot", args.expid, file_name)
        if not osp.exists(osp.join(results_path, "gp", "plot", args.expid)):
            os.makedirs(osp.join(results_path, "gp", "plot", args.expid))
    else:
        save_dir_2 = osp.join(results_path, "gp", "plot", file_name)
        if not osp.exists(osp.join(results_path, "gp", "plot")):
            os.makedirs(osp.join(results_path, "gp", "plot"))
    plt.savefig(save_dir_1)
    plt.savefig(save_dir_2)
    print(f"Evaluation Plot saved at {save_dir_1}\n")
    print(f"Evaluation Plot saved at {save_dir_2}\n")

if __name__ == '__main__':
    main()
