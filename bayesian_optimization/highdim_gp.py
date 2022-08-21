import os
import os.path as osp
import time
import torch
import tqdm
import yaml
from argparse import ArgumentParser
from attrdict import AttrDict
from torch.nn import Module

from data.highdim_gp import GPSampler
from utils.log import get_logger, RunningAverage
from utils.misc import load_module
from utils.paths import results_path, evalsets_path, datasets_path as datasets_path_


def main():
    parser = ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'eval', 'generate'], default='train')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--dimension', type=int, default=3)

    parser.add_argument('--model', default='tnpa')

    # train
    parser.add_argument('--bound', type=int, default=2)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--max_num_points', type=int, default=256)
    parser.add_argument('--min_num_points', type=int, default=64)
    parser.add_argument('--train_num_bootstrap', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=1000)

    # eval
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--eval_num_batches', type=int, default=1000)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_bootstrap', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(f'configs/gp/{args.model}.yaml', 'r') as f:
        config = yaml.safe_load(f)
        config['dim_x'] = args.dimension

    model_cls = getattr(load_module(f"models/{args.model}.py"), args.model.upper())
    model = model_cls(**config).to(device)

    root = osp.join(results_path,
                    'highdim_gp',
                    f'{args.dimension}D',
                    args.model,
                    f'min{args.min_num_points}_max{args.max_num_points}_{int(args.num_steps / 10000)}')

    if not osp.isdir(root):
        os.makedirs(root)

    with open(osp.join(root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    eval_config = {
        'eval_path': evalsets_path,
        'num_batch': args.eval_num_batches,
        'batch_size': args.eval_batch_size,
        'num_bootstrap': args.eval_num_bootstrap,
        'eval_logfile': args.eval_logfile,
        'seed': args.eval_seed
    }

    datasets_path = osp.join(datasets_path_,
                             f'gp_rbf_{args.dimension}d_bnd{args.bound}_'
                             f'batch{args.train_batch_size}'
                             f'_min{args.min_num_points}_max{args.max_num_points}')

    if args.mode == "train":
        train(
            dim_problem=args.dimension,
            model_type=args.model,
            model=model,
            root=root,
            datasets_path=datasets_path,
            bound=args.bound,
            batch_size=args.train_batch_size,
            max_num_points=args.max_num_points,
            min_num_points=args.min_num_points,
            num_bootstrap=args.train_num_bootstrap,
            learning_rate=args.lr,
            num_steps=args.num_steps,
            print_freq=args.print_freq,
            save_freq=args.save_freq,
            resume=args.resume,
            device=device
        )

    elif args.mode == "eval":
        eval(
            dim_problem=args.dimension,
            model_type=args.model,
            model=model,
            mode=args.mode,
            root=root,
            bound=args.bound,
            min_num_points=args.min_num_points,
            max_num_points=args.max_num_points,
            device=device,
            **eval_config
        )

    else:
        raise NotImplementedError


def train(
        dim_problem: int,
        model_type: str,
        model: Module,
        root: str,
        datasets_path: str,
        bound: int = 2,
        batch_size: int = 100,
        max_num_points: int = 512,
        min_num_points: int = 100,
        num_bootstrap: int = 10,
        learning_rate: float = 5e-4,
        num_steps: int = 100000,
        print_freq: int = 200,
        save_freq: int = 1000,
        resume: bool = False,
        device: torch.device = torch.device('cpu'),
):
    if not resume and not osp.isdir(datasets_path):
        os.makedirs(datasets_path)
        generate_trainset(
            path=datasets_path,
            dim_problem=dim_problem,
            bound=bound,
            num_steps=num_steps,
            batch_size=batch_size,
            max_num_points=max_num_points,
            min_num_points=min_num_points,
            device=device
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    if resume:
        # load check point
        ckpt = torch.load(osp.join(root, 'ckpt.tar'), map_location=device)
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_step = ckpt.step

    else:
        logfilename = osp.join(root, f'train_{time.strftime("%Y%m%d-%H%M")}.log')
        start_step = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not resume:
        logger.info(f"Experiment: [{model.__class__.__name__.lower()}] [{dim_problem}D]")
        logger.info(f"Device: {device}\n")
        logger.info(f"Total number of params: {sum(p.numel() for p in model.parameters())}\n")

    for step in range(start_step, num_steps + 1):
        model.train()
        optimizer.zero_grad()
        batch = torch.load(osp.join(datasets_path, f'batch{step}.tar'), map_location=device)

        if model_type in ["np", "anp", "bnp", "banp"]:
            outs = model(batch=batch,
                         num_samples=num_bootstrap)
        else:
            outs = model(batch=batch)

        outs.loss.backward()
        optimizer.step()
        scheduler.step()

        for key, val in outs.items():
            ravg.update(key, val)

        if step % print_freq == 0:
            line = f"{model.__class__.__name__.lower()}: step {step} "
            line += f"lr {optimizer.param_groups[0]['lr']:.3e} "
            line += ravg.info()
            logger.info(line)

            ravg.reset()

        if step % save_freq == 0 or step == num_steps:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.step = step + 1
            torch.save(ckpt, osp.join(root, "ckpt.tar"))

            if step in [100000, 150000, 200000, 250000, 300000]:
                torch.save(ckpt, osp.join(root, f"ckpt_{step}.tar"))


def generate_trainset(
        path,
        dim_problem: int = 5,
        bound: int = 2,
        num_steps: int = 100000,
        batch_size: int = 100,
        max_num_points: int = 512,
        min_num_points: int = 128,
        device: torch.device = torch.device('cpu'),
        seed: int = 42
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    sampler = GPSampler(dimension=dim_problem, device=device)
    for i in tqdm.tqdm(range(1, num_steps + 1), ascii=True):
        filename = f'batch{i}.tar'
        if osp.isfile(osp.join(path, filename)):
            continue

        batch = sampler(
            batch_size=batch_size,
            max_num_points=max_num_points,
            min_num_points=min_num_points,
            x_range=(-bound, bound),
            random_parameter=True
        )

        torch.save(batch, osp.join(path, filename))
        del batch

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())


def eval(
        dim_problem: int,
        model_type: str,
        model: Module,
        mode: str,
        root: str,
        eval_path: str,
        bound: int = 2,
        num_batch: int = 3000,
        batch_size: int = 16,
        min_num_points: int = 50,
        max_num_points: int = 50,
        num_bootstrap: int = 50,
        eval_logfile: str = None,
        device: torch.device = torch.device('cpu'),
        seed: int = 0
):
    if mode == "eval":
        ckpt = torch.load(osp.join(root, "ckpt.tar"), map_location=device)
        model.load_state_dict(ckpt.model)

        if eval_logfile is None:
            eval_logfile = f"eval_rbf_dim{dim_problem}.log"
        filename = osp.join(root, eval_logfile)
        logger = get_logger(filename, mode='w')

    else:
        logger = None

    path = osp.join(eval_path, 'highdim_gp')
    filename = f'rbf_dim{dim_problem}.tar'

    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        generate_evalset(
            eval_path=eval_path,
            dim_problem=dim_problem,
            bound=bound,
            num_batch=num_batch,
            batch_size=batch_size,
            min_num_points=min_num_points,
            max_num_points=max_num_points,
            device=device,
            seed=seed
        )

    eval_batches = torch.load(osp.join(path, filename), map_location=device)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    ravg = RunningAverage()
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(eval_batches, ascii=True):
            for key, val in batch.items():
                batch[key] = val.to(device)

            if model_type in ["np", "anp", "bnp", "banp"]:
                outs = model(batch=batch, num_samples=num_bootstrap)
            else:
                outs = model(batch=batch)

            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    line = f'[eval] {model.__class__.__name__.lower()}: rbf_dim{dim_problem} [loss] '
    line += ravg.info()

    if logger is not None:
        logger.info(line)

    return line


def generate_evalset(
        eval_path: str,
        dim_problem: int,
        bound: int = 2,
        num_batch: int = 3000,
        batch_size: int = 16,
        min_num_points: int = 50,
        max_num_points: int = 50,
        device: torch.device = torch.device('cpu'),
        seed: int = 0
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    sampler = GPSampler(dimension=dim_problem, device=device, seed=seed)
    batches = []
    for _ in tqdm.tqdm(range(num_batch), ascii=True):
        batches.append(
            sampler(
                batch_size=batch_size,
                min_num_points=min_num_points,
                max_num_points=max_num_points,
                x_range=(-bound, bound),
                random_parameter=True
            )
        )

    torch.manual_seed(time.time())
    torch.cuda.manual_seed(time.time())

    path = osp.join(eval_path, 'highdim_gp')
    if not osp.isdir(path):
        os.makedirs(path)

    filename = f'rbf_dim{dim_problem}.tar'
    torch.save(batches, osp.join(path, filename))


if __name__ == '__main__':
    main()
