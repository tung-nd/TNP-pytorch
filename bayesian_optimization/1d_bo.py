import os
import os.path as osp
import yaml
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

from attrdict import AttrDict
from bayeso import acquisition, covariance
from bayeso.gp import gp_kernel as gp
from bayeso.utils.utils_gp import get_prior_mu
from tqdm import tqdm
from copy import deepcopy
import argparse

from data.gp import *
from utils.misc import load_module
from utils.paths import results_path

class PlotIteration(object):
    def __init__(self, args, task_num, x, y, seed):
        self.path = osp.join(args.root, f'figures_{args.bo_mode}_{seed}')
        if not osp.exists(self.path):
            os.makedirs(self.path)

        nrows, ncols = 5, 2
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 3 * nrows),
                                           constrained_layout=True)
        self.axes = self.axes.flatten()
        self.fig.suptitle(f"{args.model.upper()} - Task {task_num}", fontsize=16)

        self.x = x
        self.y = y

    def plot(self, i, mu, sigma, obs_x, obs_y, acq, new_x, new_x_idx):
        self.axes[i].plot(self.x, self.y, color='orchid', lw=1, alpha=0.5, label='True function')

        self.axes[i].plot(self.x, mu, label="Predictive mean")
        self.axes[i].fill_between(self.x, mu - sigma, mu + sigma,
                                  color='skyblue', alpha=0.3, lw=0.0)

        self.axes[i].scatter(obs_x, obs_y, color='black', marker='o',
                             zorder=4, label="observation")

        self.axes[i].plot(self.x, acq, color="green", label="Acquisition function")

        self.axes[i].scatter(new_x, acq[new_x_idx], color='crimson', marker="v",
                             zorder=4, label="acquisition min")

        self.axes[i].grid(ls='--')
        self.axes[i].grid(ls='--')
        self.axes[i].set_xlim(-2.04, 2.04)
        self.axes[i].set_title(f"Iteration {i + 1}", fontsize=8)
        if i == 0:
            self.axes[i].legend(loc="upper right", prop={'size': 8})

    def save(self, task_num):
        plt.savefig(osp.join(self.path, f'figures{task_num}.jpeg'))
        plt.close()


def get_file(path, str_kernel, str_model, noise, seed=None):
    if noise is not None:
        str_all = f'bo_{str_kernel}_"noisy"_{str_model}'
    else:
        str_all = f'bo_{str_kernel}_{str_model}'

    if seed is not None:
        str_all += '_' + str(seed) + '.npy'
    else:
        str_all += '.npy'

    return osp.join(path, str_all)


def main():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--expid', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)  # default(-1): device="cpu"

    # Bayesian Optimization
    parser.add_argument('--bo_mode', choices=['oracle', 'models', 'plot', 'plot_bs', 'plot_v3'], default='models')
    parser.add_argument('--bo_seed', type=int, default=1)
    parser.add_argument('--num_bs', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--bo_num_bs', type=int, default=200)
    parser.add_argument('--bo_num_samples', type=int, default=200)
    parser.add_argument('--bo_num_init', type=int, default=1)
    parser.add_argument('--bo_kernel', type=str, default='rbf')
    parser.add_argument('--str_cov', choices=['matern52', 'se'], default='se')
    parser.add_argument('--bo_plot_verbose', action="store_true", default=False)
    parser.add_argument('--plot_mode', choices=['train-bs', 'bo-bs'], default='train-bs')

    # Model
    parser.add_argument('--model', type=str, default="tnpa")

    parser.add_argument('--t_noise', type=float, default=None)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # args.str_cov = 'se'
    args.num_task = 100
    args.num_iter = 100

    model = None
    if args.bo_mode == 'models':
        if args.model == '':
            raise ValueError(f"Must specify your model for mode: {args.bo_mode}")

        model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
        with open(f'configs/gp/{args.model}.yaml', 'r') as f:
            config = yaml.safe_load(f)
        if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp", "tnpa", "tnpd", "tnpnd"]:
            model = model_cls(**config)
        model.cuda()

    if 'plot' in args.bo_mode:
        args.root = osp.join(results_path, f'bayesopt_{args.bo_kernel}')
    else:
        args.expid = args.expid if args.expid is not None else ''
        args.root = osp.join(results_path, f'bayesopt_{args.bo_kernel}', args.model, args.expid)

    if not osp.isdir(args.root):
        os.makedirs(args.root)

    if args.bo_mode == 'oracle':
        oracle(args)
    elif args.bo_mode == 'models':
        models(args, model)
    elif args.bo_mode == 'plot':
        plot(args)
    else:
        raise NotImplementedError


def oracle(args):
    if args.bo_kernel == 'rbf':
        kernel = RBFKernel()
    elif args.bo_kernel == 'matern':
        kernel = Matern52Kernel()
    elif args.bo_kernel == 'periodic':
        kernel = PeriodicKernel()
    else:
        raise ValueError(f"Invalid kernel {args.bo_kernel}")

    list_dict = []
    for i_seed in tqdm(range(1, args.num_task + 1), unit='task', ascii=True):
        seed_ = args.bo_seed * i_seed

        torch.manual_seed(seed_)
        torch.cuda.manual_seed(seed_)

        sampler = GPPriorSampler(kernel, t_noise=args.t_noise)

        x = torch.linspace(-2, 2, 1000).cuda()  # (num_points,)
        x_ = x.unsqueeze(0).unsqueeze(2)  # (1, num_points, 1)

        y = sampler.sample(x_, 'cuda')  # (1, num_points, 1)
        min_y = y.min()

        batch = AttrDict()
        # random permutation of index
        idx = torch.randperm(y.shape[1])

        batch.xc = x_[:, idx[:args.bo_num_init], :]
        batch.yc = y[:, idx[:args.bo_num_init], :]

        X_train = batch.xc.squeeze(0).cpu().numpy()  # (num_init, 1)
        Y_train = batch.yc.squeeze(0).cpu().numpy()  # (num_init, 1)
        X_test = x_.squeeze(0).cpu().numpy()  # (num_points, 1)

        list_min = [batch.yc.min().cpu().numpy()]
        times = [0]
        stime = time.time()
        Plot = PlotIteration(args,
                             task_num=i_seed,
                             x=x.cpu().numpy(),
                             y=y.squeeze().cpu().numpy(),
                             seed=args.bo_seed)

        for i in range(args.num_iter):
            cov_X_X, inv_cov_X_X, hyps = gp.get_optimized_kernel(X_train, Y_train, prior_mu=None,
                                                                 str_cov=args.str_cov, fix_noise=False)

            prior_mu_train = get_prior_mu(None, X_train)  # (len(X_train), 1)
            prior_mu_test = get_prior_mu(None, X_test)  # (len(X_test), 1)

            cov_X_Xs = covariance.cov_main(args.str_cov, X_train, X_test, hyps, same_X_Xp=False)
            cov_Xs_Xs = covariance.cov_main(args.str_cov, X_test, X_test, hyps, same_X_Xp=True)
            cov_Xs_Xs = (cov_Xs_Xs + cov_Xs_Xs.T) / 2.0

            mu_ = np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), Y_train - prior_mu_train) + prior_mu_test
            Sigma_ = cov_Xs_Xs - np.dot(np.dot(cov_X_Xs.T, inv_cov_X_X), cov_X_Xs)
            sigma_ = np.expand_dims(np.sqrt(np.maximum(np.diag(Sigma_), 0.)), axis=1)

            acq_vals = -1.0 * acquisition.ei(np.ravel(mu_), np.ravel(sigma_), Y_train)
            ind_ = np.argmin(acq_vals)

            x_new = x_[:, ind_, None, :]  # (1, 1, 1)
            y_new = y[:, ind_, None, :]  # (1, 1, 1)

            if i < 10:
                Plot.plot(i, mu_.ravel(), sigma_.ravel(),
                          batch.xc[0, -1, 0].cpu(), batch.yc[0, -1, 0].cpu(),
                          acq_vals, x_new.squeeze().cpu(), ind_)

            batch.xc = torch.cat([batch.xc, x_new], dim=1)
            batch.yc = torch.cat([batch.yc, y_new], dim=1)

            X_train = batch.xc.squeeze(0).cpu().numpy()
            Y_train = batch.yc.squeeze(0).cpu().numpy()

            current_min = batch.yc.min()
            list_min.append(current_min.cpu().numpy())
            times.append(time.time() - stime)

        dict_exp = {
            'seed': seed_,
            'global': min_y.cpu().numpy(),
            'minima': np.array(list_min),
            'regrets': np.array(list_min) - min_y.cpu().numpy(),
            'times': times,
            'model': args.model,
            'cov': args.str_cov
        }
        list_dict.append(dict_exp)
        Plot.save(task_num=i_seed)

    np.save(get_file(args.root, args.bo_kernel,
                     args.model, args.t_noise, args.bo_seed), list_dict)


def models(args, model):
    if args.bo_kernel == 'rbf':
        kernel = RBFKernel()
    elif args.bo_kernel == 'matern':
        kernel = Matern52Kernel()
    elif args.bo_kernel == 'periodic':
        kernel = PeriodicKernel()
    else:
        raise ValueError(f"Invalid kernel {args.bo_kernel}")

    ckpt = torch.load(os.path.join(results_path, 'gp', args.model, args.expid, 'ckpt.tar'), map_location='cuda')
    model.load_state_dict(ckpt.model)

    list_dict = []
    for i_seed in tqdm(range(1, args.num_task + 1), unit='task', ascii=True):
        seed_ = args.bo_seed * i_seed

        torch.manual_seed(seed_)
        torch.cuda.manual_seed(seed_)

        obj_prior = GPPriorSampler(kernel, t_noise=args.t_noise)

        x = torch.linspace(-2, 2, 1000).cuda()  # (num_points,)
        x_ = x.unsqueeze(0).unsqueeze(2)  # (1, num_points, 1)

        y = obj_prior.sample(x_, device='cuda')  # (1, num_points, 1)
        min_y = y.min()

        batch = AttrDict()
        idx = torch.randperm(y.shape[1])

        batch.xc = x_[:, idx[:args.bo_num_init], :]
        batch.yc = y[:, idx[:args.bo_num_init], :]

        X_train = batch.xc.squeeze(0).cpu().numpy()  # (num_init, 1)
        Y_train = batch.yc.squeeze(0).cpu().numpy()  # (num_init, 1)

        list_min = [batch.yc.min().cpu().numpy()]
        times = [0]
        stime = time.time()
        Plot = PlotIteration(args,
                             task_num=i_seed,
                             x=x.cpu().numpy(),
                             y=y.squeeze().cpu().numpy(),
                             seed=args.bo_seed)

        model.eval()
        for i in range(0, args.num_iter):
            with torch.no_grad():
                if args.model in ["np", "anp", "bnp", "banp"]:
                    py = model.predict(xc=batch.xc,
                                       yc=batch.yc,
                                       xt=x[None, :, None],
                                       num_samples=args.bo_num_samples)
                    mu, sigma = py.mean.squeeze(0), py.scale.squeeze(0)
                else:
                    py = model.predict(xc=batch.xc, yc=batch.yc, xt=x[None, :, None])
                    mu, sigma = py.mean.squeeze(0), py.scale.squeeze(0)

            # shape: (num_samples, 1, num_points, 1)
            if mu.dim() == 4:
                var = sigma.pow(2).mean(dim=0) + mu.pow(2).mean(dim=0) - mu.mean(dim=0).pow(2)
                sigma = var.sqrt().squeeze(0)
                mu = mu.mean(dim=0).squeeze(0)

            mu_ = mu.cpu().numpy()  # (num_points, 1)
            sigma_ = sigma.cpu().numpy()  # (num_points, 1)

            # expected improvement criterion
            acq_vals = -1.0 * acquisition.ei(np.ravel(mu_), np.ravel(sigma_), Y_train)
            ind_ = np.argmin(acq_vals)

            x_new = x_[:, ind_, None, :]  # (1, 1, 1)
            y_new = y[:, ind_, None, :]  # (1, 1, 1)

            if i < 10:
                Plot.plot(i, mu_.ravel(), sigma_.ravel(),
                          batch.xc[0, -1, 0].cpu(), batch.yc[0, -1, 0].cpu(),
                          acq_vals, x_new.squeeze().cpu(), ind_)

            batch.xc = torch.cat([batch.xc, x_new], dim=1)
            batch.yc = torch.cat([batch.yc, y_new], dim=1)

            X_train = batch.xc.squeeze(0).cpu().numpy()
            Y_train = batch.yc.squeeze(0).cpu().numpy()

            current_min = batch.yc.min()
            list_min.append(current_min.cpu().numpy())
            times.append(time.time() - stime)

        dict_exp = {
            'seed': seed_,
            'global': min_y.cpu().numpy(),
            'minima': np.array(list_min),
            'regrets': np.array(list_min) - min_y.cpu().numpy(),
            'times': times,
            'model': args.model,
            'cov': args.str_cov
        }
        list_dict.append(dict_exp)
        Plot.save(task_num=i_seed)

    np.save(get_file(args.root, args.bo_kernel,
                     args.model, args.t_noise, args.bo_seed), list_dict)

def plot(args):
    all_kernels = ['rbf', 'matern', 'periodic']
    kernel_names = ['RBF', 'Matérn 5/2', 'Periodic']
    all_models = ["np", "anp", "bnp", "banp", "cnp", "canp", "tnpd", "tnpa", "tnpnd"]
    model_names = ["NP", "ANP", "BNP", "BANP", "CNP", "CANP", "TNP-D", "TNP-A", "TNP-ND"]
    colors = ['navy', 'darkgreen', 'darkgoldenrod', 'blueviolet', 'darkred', 'dimgray', 'red', 'deepskyblue', 'orange']
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    for k_id, kernel in enumerate(all_kernels):
        ax = axes[k_id]
        for i, model in enumerate(all_models):
            logfile = osp.join(results_path, f'bayesopt_{kernel}', model, f'bo_{kernel}_{model}_{args.bo_seed}.npy')
            result = np.load(logfile, allow_pickle=True)
            regrets = np.stack([result[j]['regrets'] for j in range(len(result))], axis=0)
            mean_regret = np.mean(regrets, axis=0)
            std_regret = np.std(regrets, axis=0)
            steps = np.arange(regrets.shape[1])
            
            ax.set_ylim((0.0, 0.4))
            ax.plot(steps, mean_regret, label=model_names[i], color=colors[i], lw=2.0)
            ax.fill_between(
                steps,
                mean_regret - 0.1*std_regret,
                mean_regret + 0.1*std_regret,
                alpha=0.1,
                color=colors[i])

            ax.set_facecolor('white')
            ax.grid(ls=':', color='gray', linewidth=0.5)
            ax.spines['bottom'].set_color('black')
            ax.spines['bottom'].set_linewidth('0.8')
            ax.spines['top'].set_color('black') 
            ax.spines['top'].set_linewidth('0.8') 
            ax.spines['right'].set_color('black')
            ax.spines['right'].set_linewidth('0.8')
            ax.spines['left'].set_color('black')
            ax.spines['left'].set_linewidth('0.8')

        ax.set_xlabel('Iterations', fontsize=20)
        ax.set_title(kernel_names[k_id], fontsize=20)
        
    axes[0].set_ylabel('Regret', fontsize=20)
    plt.subplots_adjust(bottom=0.24)
    fig.legend(
        labels=[name for name in model_names],
        loc="lower center", fancybox=True, shadow=True, ncol=11, fontsize=16, facecolor='white'
    )
    save_dir = osp.join(results_path, 'gp_plot.png')
    plt.savefig(save_dir, dpi=500, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    main()