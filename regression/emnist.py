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
from PIL import Image

import uncertainty_toolbox as uct

from data.image import img_to_task, task_to_img
from data.emnist import EMNIST

from utils.misc import load_module, logmeanexp
from utils.paths import results_path, evalsets_path
from utils.log import get_logger, RunningAverage

def main():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--mode',
            choices=['train', 'eval',
            'eval_all_metrics',
            'eval_all_metrics_multiple_runs',
            'plot', 'plot_samples', 'ensemble',
            'eval_multiple_runs'],
            default='train')
    parser.add_argument('--expid', type=str, default='default')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)  # default(-1): device="cpu"

    # Data
    parser.add_argument('--max_num_points', type=int, default=200)
    parser.add_argument('--class_range', type=int, nargs='*', default=[0,10])

    # Model
    parser.add_argument('--model', type=str, default="tnpd")

    # Train
    parser.add_argument('--train_seed', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--train_num_samples', type=int, default=4)
    parser.add_argument('--train_num_bs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)

    # Eval
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--eval_num_bs', type=int, default=50)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_samples', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)

    # Plot
    parser.add_argument('--plot_seed', type=int, default=1)
    parser.add_argument('--plot_num_imgs', type=int, default=16)
    parser.add_argument('--plot_num_samples', type=int, default=30)
    parser.add_argument('--plot_num_bs', type=int, default=50)
    parser.add_argument('--plot_num_ctx', type=int, default=100)
    parser.add_argument('--start_time', type=str, default=None)

    # OOD settings
    parser.add_argument('--t_noise', type=float, default=None)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.expid is not None:
        args.root = osp.join(results_path, 'emnist', args.model, args.expid)
    else:
        args.root = osp.join(results_path, 'emnist', args.model)


    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    with open(f'configs/emnist/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp", "tnpd", "tnpa", "tnpnd"]:
        model = model_cls(**config)
    model.cuda()

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        eval(args, model)
    elif args.mode == 'eval_multiple_runs':
        eval_multiple_runs(args, model)
    elif args.mode == 'eval_all_metrics':
        eval_all_metrics(args, model)
    elif args.mode == 'eval_all_metrics_multiple_runs':
        eval_all_metrics_multiple_runs(args, model)
    elif args.mode == 'plot':
        plot(args, model)
    elif args.mode == 'plot_samples':
        plot_samples(args, model)

def train(args, model):
    if osp.exists(args.root + '/ckpt.tar'):
        if args.resume is None:
            raise FileExistsError(args.root)
    else:
        os.makedirs(args.root, exist_ok=True)

    with open(osp.join(args.root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    train_ds = EMNIST(train=True, class_range=args.class_range)
    train_loader = torch.utils.data.DataLoader(train_ds,
        batch_size=args.train_batch_size,
        shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader)*args.num_epochs)

    if args.resume:
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_epoch = ckpt.epoch
    else:
        logfilename = osp.join(args.root, 'train_{}.log'.format(
            time.strftime('%Y%m%d-%H%M')))
        start_epoch = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info('Total number of parameters: {}\n'.format(
            sum(p.numel() for p in model.parameters())))

    for epoch in range(start_epoch, args.num_epochs+1):
        model.train()
        for (x, _) in tqdm(train_loader, ascii=True):
            x = x.cuda()
            batch = img_to_task(x,
                max_num_points=args.max_num_points)
            optimizer.zero_grad()

            if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp"]:
                outs = model(batch, num_samples=args.train_num_samples)
            else:
                outs = model(batch)

            outs.loss.backward()
            optimizer.step()
            scheduler.step()

            for key, val in outs.items():
                ravg.update(key, val)

        line = f'{args.model}:{args.expid} epoch {epoch} '
        line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
        line += ravg.info()
        logger.info(line)

        if epoch % args.eval_freq == 0:
            logger.info(eval(args, model) + '\n')

        ravg.reset()

        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.epoch = epoch + 1
            torch.save(ckpt, osp.join(args.root, 'ckpt.tar'))

    args.mode = 'eval'
    eval(args, model)

def gen_evalset(args):

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    eval_ds = EMNIST(train=False, class_range=args.class_range)
    eval_loader = torch.utils.data.DataLoader(eval_ds,
            batch_size=args.eval_batch_size,
            shuffle=False, num_workers=0)

    batches = []
    for x, _ in tqdm(eval_loader, ascii=True):
        batches.append(img_to_task(
            x, max_num_points=args.max_num_points,
            t_noise=args.t_noise)
        )

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path = osp.join(evalsets_path, 'emnist')
    if not osp.isdir(path):
        os.makedirs(path)

    c1, c2 = args.class_range
    filename = f'{c1}-{c2}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'

    torch.save(batches, osp.join(path, filename))

def eval(args, model):
    if args.mode == 'eval':
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        if args.eval_logfile is None:
            c1, c2 = args.class_range
            eval_logfile = f'eval_{c1}-{c2}'
            if args.t_noise is not None:
                eval_logfile += f'_{args.t_noise}'
            eval_logfile += '.log'
        else:
            eval_logfile = args.eval_logfile
        filename = osp.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    path = osp.join(evalsets_path, 'emnist')
    c1, c2 = args.class_range
    filename = f'{c1}-{c2}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)

    eval_batches = torch.load(osp.join(path, filename))

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

    c1, c2 = args.class_range
    line = f'{args.model}:{args.expid} {c1}-{c2} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    line += ravg.info()

    if logger is not None:
        logger.info(line)

    return line

def eval_multiple_runs(args, model):
    num_runs = 5

    path = osp.join(evalsets_path, 'emnist')
    c1, c2 = args.class_range
    filename = f'{c1}-{c2}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'
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
            ckpt = torch.load(osp.join(results_path, 'emnist', args.model, f'run{i+1}', 'ckpt.tar'), map_location='cuda')
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
    
    line = f'{args.model}:{c1}-{c2} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    line += f'\n'

    for i in range(num_runs):
        line += f'run {i+1}: tar_ll {tar_ll_all[i]}'
        line += '\n'

    line += f'mean and std of nll of {num_runs} runs:\n'
    line += f'{np.mean(all_nll)} +- {np.std(all_nll)}'

    filename = f'eval_{c1}-{c2}_{num_runs}runs'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.log'
    logger = get_logger(osp.join(results_path, 'emnist', args.model, filename), mode='w')

    logger.info(line)

def eval_all_metrics(args, model):
    ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'), map_location='cuda')
    model.load_state_dict(ckpt.model)

    path = osp.join(evalsets_path, 'emnist')
    c1, c2 = args.class_range
    if not osp.isdir(path):
        os.makedirs(path)
    filename = f'{c1}-{c2}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)

    eval_batches = torch.load(osp.join(path, filename))

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    model.eval()
    with torch.no_grad():
        ravgs = [RunningAverage() for _ in range(3)] # 3 types of metrics
        for batch in tqdm(eval_batches, ascii=True):
            for key, val in batch.items():
                batch[key] = val.cuda()

            if args.model in ["np", "anp", "bnp", "banp"]:
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
                var = std.pow(2).mean(dim=0) + mean.pow(2).mean(dim=0) - mean.mean(dim=0).pow(2)
                std = var.sqrt().squeeze(0)
                mean = mean.mean(dim=0).squeeze(0)
            
            mean, std = mean.squeeze().cpu().numpy().flatten(), std.squeeze().cpu().numpy().flatten()
            yt = batch.yt.squeeze().cpu().numpy().flatten()

            acc = uct.metrics.get_all_accuracy_metrics(mean, yt, verbose=False)
            sharpness = uct.metrics.get_all_sharpness_metrics(std, verbose=False)
            scoring_rule = {'tar_ll': ll.tar_ll.item()}

            batch_metrics = [acc, sharpness, scoring_rule]
            for i in range(len(batch_metrics)):
                ravg, batch_metric = ravgs[i], batch_metrics[i]
                for k in batch_metric.keys():
                    ravg.update(k, batch_metric[k])

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f'{args.model}:{args.expid}:{c1}-{c2} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    
    line += '\n'

    for ravg in ravgs:
        line += ravg.info()
        line += '\n'

    filename = f'eval_{c1}-{c2}_all_metrics'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.log'
    logger = get_logger(osp.join(results_path, 'emnist', args.model, args.expid, filename), mode='w')
    logger.info(line)

    return line

def eval_all_metrics_multiple_runs(args, model):
    num_runs = 5

    path = osp.join(evalsets_path, 'emnist')
    c1, c2 = args.class_range
    if not osp.isdir(path):
        os.makedirs(path)
    filename = f'{c1}-{c2}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.tar'
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)

    eval_batches = torch.load(osp.join(path, filename))

    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed(args.eval_seed)

    all_metrics = [{}, {}, {}] # acc, sharpness, likelihood

    with torch.no_grad():
        for i in range(num_runs):
            model_ = deepcopy(model)
            ckpt = torch.load(osp.join(results_path, 'emnist', args.model, f'run{i+1}', 'ckpt.tar'), map_location='cuda')
            model_.load_state_dict(ckpt['model'])
            model_.eval()

            print ('Evaluating run %d' % (i+1))

            ragvs = [RunningAverage() for _ in range(3)] # 3 types of metrics

            for batch in tqdm(eval_batches, ascii=True):
                for key, val in batch.items():
                    batch[key] = val.cuda()
                
                if args.model in ["np", "anp", "bnp", "banp"]:
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

                if mean.dim() == 4:
                    var = std.pow(2).mean(dim=0) + mean.pow(2).mean(dim=0) - mean.mean(dim=0).pow(2)
                    std = var.sqrt().squeeze(0)
                    mean = mean.mean(dim=0).squeeze(0)
                
                mean, std = mean.squeeze().cpu().numpy().flatten(), std.squeeze().cpu().numpy().flatten()
                yt = batch.yt.squeeze().cpu().numpy().flatten()

                acc = uct.metrics.get_all_accuracy_metrics(mean, yt, verbose=False)
                sharpness = uct.metrics.get_all_sharpness_metrics(std, verbose=False)
                scoring_rule = {'tar_ll': ll.tar_ll.item()}

                batch_metrics = [acc, sharpness, scoring_rule]
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

    line = f'{args.model}:{c1}-{c2} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    line += f'\n'
    line += f'nll of {num_runs} runs:\n'

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

    filename = f'eval_{c1}-{c2}_all_metrics_{num_runs}runs'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += '.log'
    logger = get_logger(osp.join(results_path, 'emnist', args.model, filename), mode='w')

    logger.info(line)

def plot(args, model):
    if args.mode == 'plot':
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)

    eval_ds = EMNIST(train=False, class_range=args.class_range)
    torch.manual_seed(args.plot_seed)
    rand_ids = torch.randperm(len(eval_ds))[:args.plot_num_imgs]
    test_data = [eval_ds[i][0] for i in rand_ids]
    test_data = torch.stack(test_data, dim=0).cuda()
    batch = img_to_task(test_data, max_num_points=None, num_ctx=args.plot_num_ctx, target_all=True)
    
    model.eval()
    with torch.no_grad():
        if args.model in ["np", "anp", "bnp", "banp", "tnpa", "tnpnd"]:
            outs = model.predict(batch.xc, batch.yc, batch.xt, num_samples=args.eval_num_samples)
        else:
            outs = model.predict(batch.xc, batch.yc, batch.xt)

    if args.model not in ["neubanp"]:
        mean, std = outs.loc, outs.scale
    else:
        mean, std = outs.mean, outs.std

    # shape: (num_samples, 1, num_points, 1)
    if mean.dim() == 4:
        mean = mean.mean(dim=0)

    task_img, completed_img = task_to_img(batch.xc, batch.yc, batch.xt, mean, shape=(1,28,28))
    _, orig_img = task_to_img(batch.xc, batch.yc, batch.xt, batch.yt, shape=(1,28,28))

    task_img = (task_img * 255).int().cpu().numpy().transpose(0,2,3,1)
    completed_img = (completed_img * 255).int().cpu().numpy().transpose(0,2,3,1)
    orig_img = (orig_img * 255).int().cpu().numpy().transpose(0,2,3,1)

    c1, c2 = args.class_range
    save_dir = osp.join(args.root, f'plots_{c1}-{c2}')
    os.makedirs(save_dir, exist_ok=True)

    for i in range(args.plot_num_imgs):
        Image.fromarray(orig_img[i].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_orig.jpg' % (i+1))
        Image.fromarray(task_img[i].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_task.jpg' % (i+1))
        Image.fromarray(completed_img[i].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_completed.jpg' % (i+1))

def plot_samples(args, model):
    if args.mode == 'plot_samples':
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'))
        model.load_state_dict(ckpt.model)

    eval_ds = EMNIST(train=False, class_range=args.class_range)
    torch.manual_seed(args.plot_seed)
    rand_ids = torch.randperm(len(eval_ds))[:args.plot_num_imgs]
    test_data = [eval_ds[i][0] for i in rand_ids]
    test_data = torch.stack(test_data, dim=0).cuda()
    
    list_num_ctx = [10, 20, 50, 100, 150]
    batches = [img_to_task(test_data, max_num_points=None, num_ctx=i, target_all=True) for i in list_num_ctx]
    all_samples = []
    
    model.eval()
    with torch.no_grad():
        for batch in batches:
            if args.model in ["np", "anp", "bnp", "banp", "tnpa", "tnpnd"]:
                samples = model.sample(batch.xc, batch.yc, batch.xt, num_samples=args.eval_num_samples)
            else:
                samples = model.sample(batch.xc, batch.yc, batch.xt)
            all_samples.append(samples)

    c1, c2 = args.class_range
    save_dir = osp.join(args.root, f'sample_plots_{c1}-{c2}')
    os.makedirs(save_dir, exist_ok=True)

    # save original images
    _, orig_img = task_to_img(batches[-1].xc, batches[-1].yc, batches[-1].xt, batches[-1].yt, shape=(1,28,28)) # (num_imgs, 32, 32, 3)
    orig_img = (orig_img * 255).int().cpu().numpy().transpose(0,2,3,1)
    for i in range(args.plot_num_imgs):
        Image.fromarray(orig_img[i].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_orig.jpg' % (i+1))
    
    for i in range(len(list_num_ctx)):
        num_ctx = list_num_ctx[i]
        batch = batches[i]
        samples = all_samples[i]

        for j in range(args.eval_num_samples):
            task_img, completed_img = task_to_img(batch.xc, batch.yc, batch.xt, samples[j], shape=(1,28,28)) # (num_imgs, 32, 32, 3)

            task_img = (task_img * 255).int().cpu().numpy().transpose(0,2,3,1)
            completed_img = (completed_img * 255).int().cpu().numpy().transpose(0,2,3,1)

            for k in range(args.plot_num_imgs):
                Image.fromarray(task_img[k].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_task_%d_ctx.jpg' % (k+1, num_ctx))
                Image.fromarray(completed_img[k].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_completed_%d_ctx_%d_samples.jpg' % (k+1, num_ctx, j+1))

if __name__ == '__main__':
    main()
