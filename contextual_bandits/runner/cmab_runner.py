import os
import os.path as osp
import numpy as np
import yaml
import torch
import time
import matplotlib.pyplot as plt

from attrdict import AttrDict
from tqdm import tqdm

from data.wheel import sample_wheel_data, WheelBanditSampler
from utils.misc import load_module
from utils.log import get_logger, RunningAverage, plot_log
from runner import evalsets_path, results_path, datasets_path


def cmab(args):
    args.expconfig = args.expid or "default"
    args.cmab_models = "models_" + args.model + ".yaml"
    device = args.device

    if args.cmab_mode == 'train':
        name = args.model
        model_cls = getattr(load_module(f'models/{name}.py'), name.upper())  # ex. from models.cnp import CNP
        with open(osp.join("configs", f"{args.cmab_data}", f"{name}.yaml")) as g:
            config = yaml.safe_load(g)
        model = model_cls(**config).to(device)
        model.train()
        path, filename = get_train_path(args)
        file = osp.join(path, filename)
        if osp.exists(file):
            if args.resume is None:
                raise FileExistsError(file)
        else:
            os.makedirs(path, exist_ok=True)
        args.root = path
        with open(osp.join(args.root, 'args.yaml'), 'w') as f:
            yaml.dump(args.__dict__, f)
        train(args, model)

    if args.cmab_mode == "eval":
        args.num_contexts = 2000
        path, filename = get_train_path(args)
        name = args.model
        if args.model != "uniform":
            model_cls = getattr(load_module(f'models/{name}.py'), name.upper())  # ex. from models.cnp import CNP
            with open(osp.join("configs", f"{args.cmab_data}", f"{name}.yaml")) as g:
                config = yaml.safe_load(g)
            model = model_cls(**config).to(device)
            model.eval()
            file = osp.join(path, filename)
            if not osp.exists(file):
                raise FileNotFoundError(file)
            else:
                ckpt = torch.load(file)
                model.load_state_dict(ckpt.model)

        for i in range(args.cmab_eval_seed_start, args.cmab_eval_seed_end + 1):
            args.cmab_eval_seed = i
            if args.model == "uniform":
                actor = UNIFORM()
            else:
                actor = DummyActor(model, device=args.device, method=args.cmab_eval_method)
            path, filename = get_eval_path(args)
            with open(osp.join(path, 'args.yaml'), 'w') as f:
                yaml.dump(args.__dict__, f)
            eval(args, [actor])

        args.model = "uniform"
        for i in range(args.cmab_eval_seed_start, args.cmab_eval_seed_end + 1):
            args.cmab_eval_seed = i
            if args.model == "uniform":
                actor = UNIFORM()
            else:
                actor = DummyActor(model, device=args.device, method=args.cmab_eval_method)
            path, filename = get_eval_path(args)
            with open(osp.join(path, 'args.yaml'), 'w') as f:
                yaml.dump(args.__dict__, f)
            eval(args, [actor])

        args.expid = args.expid or "default"
        if args.expid is None:
            raise ValueError("Must specify expid for plotting")
        args.num_contexts = 2000

        names = []
        with open(osp.join("configs", f"{args.cmab_data}", args.cmab_models)) as f:
            f = yaml.safe_load(f)
            for name in f:
                names.append(name)
        plot(args, names)


def get_bandit_dataset(args):
    if args.cmab_mode == "train":
        path, filename = get_trainset_path(args)
        if not osp.isfile(osp.join(path, f"{filename}.tar")):
            gen_trainset(args)
    else:
        path, filename = get_evalset_path(args)
        if not osp.isfile(osp.join(path, f"{filename}.tar")):
            gen_evalset(args)
    dataset = torch.load(os.path.join(path, f"{filename}.tar"))
    return dataset


def gen_trainset(args):

    print(f"Generating {args.cmab_data} bandit training sets...")

    _f = {
        "wheel": WheelBanditSampler,
    }
    sampler = _f[args.cmab_data]()
    seed = 0
    batches = []
    for i in tqdm(range(args.cmab_train_num_batches), ascii=True):
        seed = i * args.cmab_train_batch_size
        batches.append(sampler.sample(
            batch_size=args.cmab_train_batch_size,
            device=args.device,
            seed=seed,
            reward=args.cmab_train_reward
        ))

    path, filename = get_trainset_path(args)
    torch.save(batches, osp.join(path, f"{filename}.tar"))


def gen_evalset(args):

    print(f"Generating {args.cmab_data} evaluation sets...")

    if args.cmab_data == "wheel":
        sample_vals = sample_wheel_data(delta=args.cmab_wheel_delta, num_contexts=args.cmab_eval_num_contexts, seed=args.cmab_eval_seed)

    path, filename = get_evalset_path(args)
    torch.save(sample_vals, osp.join(path, f"{filename}.tar"))


def get_trainset_path(args):
    path = osp.join(datasets_path, args.cmab_data, f"trainset-{args.cmab_train_reward}-R")
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)
    if args.model == "uniform":
        filename = "none"
    else:
        filename = f"S{args.cmab_train_seed}-B{args.cmab_train_batch_size}"
    if args.cmab_train_num_batches != 1:
        filename += f"x{args.cmab_train_num_batches}"

    return path, filename


def get_evalset_path(args):
    path = osp.join(evalsets_path, args.cmab_data, "evalset")
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)
    filename = f'S{args.cmab_eval_seed}-C{args.cmab_eval_num_contexts}-d{args.cmab_wheel_delta}'
    return path, filename


def get_train_path(args):
    _, folder = get_trainset_path(args)
    if args.cmab_train_update_freq != 1:
        folder += f"-uf{args.cmab_train_update_freq}"
    path = osp.join(results_path, args.cmab_data, f"train-{args.cmab_train_reward}-R", args.model, folder, args.expconfig)
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)
    filename = 'ckpt.tar'
    return path, filename


def get_eval_path(args):
    _, folder = get_trainset_path(args)
    path = osp.join(results_path, args.cmab_data, f"eval-{args.cmab_train_reward}-R", args.model, folder, args.expconfig)
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)
    filename = f"S{args.cmab_eval_seed}-C{args.cmab_eval_num_contexts}-d{args.cmab_wheel_delta}"
    return path, filename


def get_plot_path(args):
    path = osp.join(results_path, args.cmab_data, f"plot-{args.cmab_train_reward}-R", args.expid)
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)
    filename = f"S{args.cmab_plot_seed_start}-{args.cmab_plot_seed_end}-C{args.cmab_eval_num_contexts}-d{args.cmab_wheel_delta}-{args.cmab_eval_method}"
    return path, filename


def train(args, model):
    torch.manual_seed(args.cmab_train_seed)
    torch.cuda.manual_seed(args.cmab_train_seed)

    dataset = get_bandit_dataset(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.num_epochs / args.cmab_train_update_freq))
    device = args.device

    if args.resume:
        ckpt = torch.load(os.path.join(args.root, f'ckpt.tar'))
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_step = ckpt.step
    else:
        args.start_time = time.strftime("%Y%m%d-%H%M")
        logfilename = os.path.join(args.root, f'train_{args.start_time}.log')
        start_step = 1
    if os.path.exists(logfilename):
        if not args.resume:
            os.remove(logfilename)

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    logger.info(f"Experiment: Bandit Train | {args.expid}")
    logger.info(f"Device: {device}\n")
    logger.info(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}\n')

    for step in range(start_step, args.num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        for batch in dataset:
            for k, v in batch.items():
                if v is not None:
                    batch[k] = v.to(device)
            
            if args.model in ["bnp", "banp"]:
                outs = model(batch, 4)
            elif args.model in ["np", "anp"]:
                outs = model(batch, 1)
            else:
                outs = model(batch)

            if step % args.cmab_train_update_freq == 0:
                outs.loss.backward()
                optimizer.step()
                scheduler.step()

            for key, val in outs.items():
                ravg.update(key, val)

        if step % args.print_freq == 0:
            _, filename = get_trainset_path(args)
            line = f'[model] {model._get_name()}-{filename} [step] {step} '
            line += f'[lr] {optimizer.param_groups[0]["lr"]:.3e} '
            line += f"[train_loss] "
            line += ravg.info()
            logger.info(line)
            ravg.reset()

        if step % args.save_freq == 0 or step == args.num_epochs:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.step = step + 1
            if not osp.exists(args.root):
                os.makedirs(args.root, exist_ok=True)
            torch.save(ckpt, os.path.join(args.root, f'ckpt.tar'))


    plot_log(logfilename)
    if args.num_epochs >= 50000:
        plot_log(logfilename, 0, 50000)
        plot_log(logfilename, 50000, args.num_epochs)


def eval(args, models):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    _dataset = get_bandit_dataset(args)
    dataset, opt_rewards, opt_actions, num_actions, context_dim = _dataset

    t_init = time.time()
    _results = run_contextual_bandit(context_dim, num_actions, dataset, models, args.cmab_num_bs, args.device)
    h_actions, h_rewards = _results

    path, filename = get_eval_path(args)
    _, folder = get_trainset_path(args)
    folder += f"-{args.cmab_eval_method}"
    path = osp.join(path, folder)
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)

    freq, duration = log_results(args, models, opt_rewards, opt_actions, h_rewards, t_init)
    results = [[model.name for model in models], h_actions, h_rewards, opt_actions, opt_rewards, freq, duration]

    np.save(osp.join(path, filename), results, allow_pickle=True)


def plot(args, names):
    rewards = []
    regrets = []
    for name in names:
        args.model = name
        _rewards = []
        _regrets = []
        for i in tqdm(range(args.cmab_eval_seed_start, args.cmab_eval_seed_end + 1), ascii=True):
            args.cmab_eval_seed = i
            path, filename = get_eval_path(args)
            _, folder = get_trainset_path(args)
            if name == "uniform":
                folder = f"none"
                if args.cmab_train_num_batches != 1:
                    folder += f"x{args.cmab_train_num_batches}"

            folder += f"-{args.cmab_eval_method}"
            results = np.load(osp.join(path, folder, f"{filename}.npy"), allow_pickle=True)
            dataset, a, r, opt_a, opt_r, time, freq = results  # [N,1], [N,]
            _rewards.append(r[:, 0])  # [N,]
            _regrets.append(opt_r - r[:, 0])  # [N,]
        _rewards = np.vstack(_rewards)  # [B,N]
        _regrets = np.vstack(_regrets)  # [B,N]
        rewards.append(_rewards)
        regrets.append(_regrets)

    # rewards = np.stack(rewards, -1)  # [B,N,Nm]
    # cum_rewards = np.cumsum(rewards, 1)  # [B,N,Nm]
    regrets = np.stack(regrets, -1)  # [B,N,Nm]
    cum_regrets = np.cumsum(regrets, 1)  # [B,N,Nm]

    _plot_cum_reg(args, names, cum_regrets)
    _log(args, names, regrets)


def _log(args, names, values):
    cum_values = np.cumsum(values, 1)
    path, filename = get_eval_path(args)
    path = osp.join(path, filename)
    file = f"regret.log"
    with open(osp.join(path, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)
    logger = get_logger(osp.join(path, file))
    line = f"{filename}\n\n"

    line += "[cumulative regret]\n\n"
    for j, name in enumerate(names):
        v = cum_values[:, -1, j]  # [B,]
        if j == 0:
            base_cum = np.mean(v, 0)
        v = v / base_cum * 100
        mu, sigma = np.mean(v, 0), np.std(v, 0)
        line += f"{name} \n\n{mu:.2f} +-{sigma:.2f} \n{mu: .2f} +-{sigma: .2f} \n\n"

    line += "[simple regret]\n\n"
    for j, name in enumerate(names):
        v = values[:, -500:, j].mean(0) # [B,]
        if j == 0:
            base_sim = np.mean(v, 0)
        v = v / base_sim * 100
        mu, sigma = np.mean(v, 0), np.std(v, 0)
        line += f"{name} \n\n{mu:.2f} +-{sigma:.2f} \n{mu: .2f} +-{sigma: .2f} \n\n"

    logger.info(line)


def _plot_cum_reg(args, names, values):
    mu, sigma = np.mean(values, 0), np.std(values, 0)  # [N,Nm]
    path, filename = get_eval_path(args)
    path = osp.join(path, filename)
    os.makedirs(path, exist_ok=True)
    np.save(osp.join(path, f"cumulative-regret.npy"), [mu, sigma], allow_pickle=True)
    x_axis = np.array(range(args.cmab_eval_num_contexts))  # steps

    plt.clf()
    for j, name in enumerate(names):
        plt.plot(x_axis, mu[:, j], label=f"{name}")
        plt.fill_between(x_axis, mu[:, j] - sigma[:, j], mu[:, j] + sigma[:, j],
                         alpha=0.2, linewidth=0.0)
    plt.legend()
    plt.xlabel("steps")
    plt.ylabel(f"cumulative-regret")
    plt.title(f"Wheel Bandit - cumulative-regret")
    plt.savefig(osp.join(path, f"{filename}.jpg"))


def run_contextual_bandit(context_dim, num_actions, dataset, models, num_bs, device):
    """Run a contextual bandit problem on a set of algorithms.
    Args:
      context_dim: Dimension of the context.
      num_actions: Number of available actions.
      dataset: Matrix where every row is a context + num_actions rewards.
      models: List of algorithms to use in the contextual bandit instance.
      num_bs
      device
    Returns:
      h_actions: Matrix with actions: size (num_context, num_algorithms).
      h_rewards: Matrix with rewards: size (num_context, num_algorithms).
    """
    # Nm : len(models), number of models to evaluate
    # N : number of steps = number of (context, action, reward) = number of context
    num_contexts = dataset.shape[0]  # N

    # Create contextual multi-armed bandit (wheel)
    cmab = ContextualBandit(context_dim, num_actions)  # 2, 5
    cmab.feed_data(dataset)

    h_actions = None
    h_rewards = None

    # Run the contextual bandit process
    for i in tqdm(range(num_contexts), ascii=True):
        context = cmab.context(i)  # [2,]
        actions = [model.action(context, num_bs) for model in models]  # [Nm,]
        rewards = [cmab.reward(i, action) for action in actions]  # [Nm,]

        for j, model in enumerate(models):
            model.update(context, actions[j], rewards[j])
        if h_actions is None:
            h_actions = np.array([actions])  # [1,Nm]
            h_rewards = np.array([rewards])  # [1,Nm]
        else:
            h_actions = np.vstack((h_actions, np.array([actions])))  # [H,Nm]
            h_rewards = np.vstack((h_rewards, np.array([rewards])))  # [H,Nm]

    return h_actions, h_rewards  # [N,Nm], [N,Nm]


def log_results(args, models, opt_rewards, opt_actions, h_rewards, t_init):
    """Logs summary statistics of the performance of each algorithm."""

    name = args.cmab_data
    path, filename = get_eval_path(args)
    _, folder = get_trainset_path(args)
    file = osp.join(path, f"{folder}-{args.cmab_eval_method}", f"{filename}.log")
    logger = get_logger(file)

    duration = time.time() - t_init
    line = "\n"
    line += '---------------------------------------------------\n'
    line += '---------------------------------------------------\n'
    line += f'{name} bandit completed after {duration} seconds.\n'
    line += '---------------------------------------------------\n'

    performance_pairs = []
    for j, model in enumerate(models):
        performance_pairs.append((model.name, np.sum(h_rewards[:, j])))
    performance_pairs = sorted(performance_pairs,
                               key=lambda elt: elt[1],
                               reverse=True)
    for i, (name, reward) in enumerate(performance_pairs):
        line += f'{i:3}) {name:20}| \t \t total reward = {reward:10}.\n'

    line += '---------------------------------------------------\n'
    line += f'Optimal total reward = {np.sum(opt_rewards)}.\n'
    line += f'Total Steps = {opt_rewards.shape[0]}.\n'
    line += 'Frequency of optimal actions (action, frequency):\n'
    freq = [[elt, list(opt_actions).count(elt)] for elt in set(opt_actions)]
    line += f"{freq}\n"
    line += '---------------------------------------------------\n'
    line += '---------------------------------------------------\n'

    logger.info(line)

    return freq, duration


class ContextualBandit(object):
    """Implements a Contextual Bandit with d-dimensional contexts and k arms."""

    def __init__(self, context_dim, num_actions):
        """Creates a contextual bandit object.
        Args:
          context_dim: Dimension of the contexts.
          num_actions: Number of arms for the multi-armed bandit.
        """

        self._context_dim = context_dim
        self._num_actions = num_actions

    def feed_data(self, data):
        """Feeds the data (contexts + rewards) to the bandit object.
        Args:
          data: Numpy array with shape [n, d+k], where n is the number of contexts,
            d is the dimension of each context, and k the number of arms (rewards).
        Raises:
          ValueError: when data dimensions do not correspond to the object values.
        """

        if data.shape[1] != self.context_dim + self.num_actions:
            raise ValueError('Data dimensions do not match.')

        self._number_contexts = data.shape[0]
        self.data = data
        self.order = range(self.number_contexts)

    def reset(self):
        """Randomly shuffle the order of the contexts to deliver."""
        self.order = np.random.permutation(self.number_contexts)

    def context(self, number):
        """Returns the number-th context."""
        return self.data[self.order[number]][:self.context_dim]

    def reward(self, number, action):
        """Returns the reward for the number-th context and action."""
        return self.data[self.order[number]][self.context_dim + action]

    def optimal(self, number):
        """Returns the optimal action (in hindsight) for the number-th context."""
        return np.argmax(self.data[self.order[number]][self.context_dim:])

    @property
    def context_dim(self):
        return self._context_dim

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def number_contexts(self):
        return self._number_contexts


class DummyActor():
    def __init__(self, model, num_actions=5, device=None, method="mean"):
        self.model = model
        self.name = model._get_name()
        self.Na = num_actions
        self.hc = None
        self.hr = None
        self.device = device
        self.method = method

    def action(self, context, num_bs=10):  # context [Dx=2]
        if self.hc is None:
            action = np.random.randint(self.Na)
        else:
            mu, sigma = self.infer(context, num_bs)

            """
            UCB with sampled function estimate
            """
            if self.method == "mean":
                criterion = mu.cpu().detach().numpy()[0][0]
            elif self.method == "ucb":
                criterion = mu.cpu().detach().numpy()[0][0] + sigma.cpu().detach().numpy()[0][0]  # [Dy=5,]
            elif self.method == "perturb":
                dist = torch.distributions.Normal(mu[0][0], sigma[0][0])  # [Dy=5,]
                criterion = dist.sample().detach().cpu().numpy()  # [Dy=5,]
            action = np.argmax(np.array(criterion))  # scalar

        return action  # scalar

    def infer(self, context, num_bs):
        xc = torch.from_numpy(self.hc).to(self.device).type(torch.float32).unsqueeze(0)  # [B=1,H,Dx=2]
        yc = torch.from_numpy(self.hr).to(self.device).type(torch.float32).unsqueeze(0)  # [B=1,H,Dy=5]
        xt = torch.from_numpy(context).to(self.device).type(torch.float32).reshape(1, 1, 2)  # [B=1,Nt=1,Dx=2]

        py = self.model.predict(xc, yc, xt)
        mu, sigma = py.loc, py.scale  # [B=1,Nt=1,Dy=5]
        return mu, sigma

    def update(self, context, action, reward):
        """
        update the history of chosen action
        """
        a = int(action)
        c = context.reshape(1,-1)  # [1,2]
        r = np.random.normal(size=5).reshape(1, -1)  # [1,5]
        r[0, a] = reward

        if self.hc is None:
            self.hc = c  # [1,2]
            self.hr = r  # [1,5]
        else:
            self.hc = np.vstack([self.hc, c])  # [H,2]
            self.hr = np.vstack([self.hr, r])  # [H,5]


class UNIFORM():
    def __init__(self, num_actions=5):
        self.name = "Uniform"
        self.Na = num_actions

    def action(self, context, num_bs=10, device=None):
        action = np.random.randint(self.Na)
        return action

    def update(self, context, action, reward):
        pass