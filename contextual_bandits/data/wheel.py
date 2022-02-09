import torch
import numpy as np

from attrdict import AttrDict
from utils.misc import one_hot
from torch.utils.data import Dataset

def dummy(x, idx, seed=0):  # x [...,N,I]  idx [N,]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    _x = torch.randn(size=x.size())
    for n, i in enumerate(idx):
        _x[..., n, i] = x[..., n, i]  # retain only values corresponding to idx
    return _x  # [...,N,I]

class WheelBanditDataset(Dataset):
  def __init__(self, batch):
    self.batch = batch

  def __len__(self):
    return self.batch.x.shape[0]

  def __getitem__(self, index):
    batch = AttrDict()
    for k in self.batch.keys():
      if self.batch[k] is not None:
        batch[k] = self.batch[k][index]
      # else:
      #   batch[k] = None
    return batch

class WheelBanditSampler():
    def __init__(self):
        pass

    def sample(self, batch_size=8, num_contexts=512, num_targets=50, device=None, seed=0, reward="optimal"):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        deltas = torch.rand(batch_size)

        Nc = num_contexts
        Nt = num_targets
        N = Nc + Nt  # num_points
        x, xc, xt, y, yc, yt, idx, d = [], [], [], [], [], [], [], []

        batch = AttrDict()
        for i, delta in enumerate(deltas):
            dataset, opt_rewards, opt_actions, _, _ = sample_wheel_data(N, delta, seed+i)  # [N,7], [N,], [N,]
            contexts = torch.from_numpy(dataset[:, :2], ).type(torch.float32)  # [N,2]
            rewards = torch.from_numpy(dataset[:, 2:], ).type(torch.float32)  # [N,5]

            _x = contexts.to(device)  # [N,2]
            _y = rewards.to(device)  # [N,5]

            if reward != "all":
                if reward == "optimal":
                    sampled_actions = opt_actions  # index of actions to random sample
                elif reward == "random":
                    sampled_actions = np.random.randint(0, 5, N)  # [N,]

                _y = dummy(_y, sampled_actions, seed + i).to(device)  # [N,5]
                idx.append(torch.from_numpy(sampled_actions))  # [N,]

            x.append(_x)  # [N,2]
            xc.append(_x[:Nc, :])  # [Nc,2]
            xt.append(_x[Nc:, :])  # [Nt,2]
            y.append(_y)  # [N,5]
            yc.append(_y[:Nc, :])  # [Nc,5]
            yt.append(_y[Nc:, :])  # [Nt,5]
            d.append(delta)  # [1,]

        batch.x = torch.stack(x, 0)  # [B,N,2]
        batch.xc = torch.stack(xc, 0)  # [B,Nc,2]
        batch.xt = torch.stack(xt, 0)  # [B,Nt,2]
        batch.y = torch.stack(y, 0)  # [B,N,5]
        batch.yc = torch.stack(yc, 0)  # [B,Nc,5]
        batch.yt = torch.stack(yt, 0)  # [B,Nt,5]
        if reward == "all":
            batch.w = None
        else:
            batch.w = one_hot(torch.stack(idx, 0), batch.y.size(-1))  # [B,N,Dy]
        batch.d = torch.tensor(d)  # [B,]

        return batch


def sample_wheel_data(num_contexts=2000, delta=0.95, seed=0):
    num_actions = 5
    context_dim = 2
    mean_v = [1.2, 1.0, 1.0, 1.0, 1.0]
    std_v = [0.05, 0.05, 0.05, 0.05, 0.05]
    mu_large = 50
    std_large = 0.01

    dataset, opt_wheel = sample_wheel_bandit_data(num_contexts, delta,
                                                  mean_v, std_v,
                                                  mu_large, std_large,
                                                  seed=seed)
    opt_rewards, opt_actions = opt_wheel

    return dataset, opt_rewards, opt_actions, num_actions, context_dim  # [N,7], [N,], [N,], 5, 2


def sample_wheel_bandit_data(num_contexts, delta, mean_v, std_v,
                             mu_large, std_large, seed=0):
    """Samples from Wheel bandit game (see https://arxiv.org/abs/1802.09127).
    Args:
      num_contexts: Number of points to sample, i.e. (context, action, rewards).
      delta: Exploration parameter: high reward in one region if norm above delta.
      mean_v: Mean reward for each action if context norm is below delta.
      std_v: Gaussian reward std for each action if context norm is below delta.
      mu_large: Mean reward for optimal action if context norm is above delta.
      std_large: Reward std for optimal action if context norm is above delta.
    Returns:
      dataset: Sampled matrix with n rows: (context, action, rewards).
      opt_vals: Vector of expected optimal (reward, action) for each context.
    """
    np.random.seed(0)

    context_dim = 2
    num_actions = 5

    data = []
    rewards = []
    opt_actions = []
    opt_rewards = []

    # sample uniform contexts in unit ball
    while len(data) < num_contexts:  # num_contexts N
      raw_data = np.random.uniform(-1, 1, (int(num_contexts / 3), context_dim))

      for i in range(raw_data.shape[0]):
        if np.linalg.norm(raw_data[i, :]) <= 1:
          data.append(raw_data[i, :])

    contexts = np.stack(data)[:num_contexts, :]  # [N,2]

    # sample rewards
    for i in range(num_contexts):
      r = [np.random.normal(mean_v[j], std_v[j]) for j in range(num_actions)]
      if np.linalg.norm(contexts[i, :]) >= delta:
        """
        outer part optimal: k 사분면 -> action k
        """
        r_big = np.random.normal(mu_large, std_large)

        if contexts[i, 1] > 0:
          if contexts[i, 0] > 0:
            r[1] = r_big
            opt_actions.append(1)  # action 1, +x +y
          else:
            r[2] = r_big
            opt_actions.append(2)  # action 2, -x +y
        else:
          if contexts[i, 0] <= 0:
            r[3] = r_big
            opt_actions.append(3)  # action 3, -x -y
          else:
            r[4] = r_big
            opt_actions.append(4)  # action 4, +x -y
      else:
        """
        inner part optimal: action 0 in this setting
        """
        opt_actions.append(np.argmax(mean_v))  # action 0, always

      opt_rewards.append(r[opt_actions[-1]])
      rewards.append(r)

    rewards = np.stack(rewards)  # [N,5]
    dataset = np.hstack((contexts, rewards))  # [N,7]  hstack: 2차원에서 concatenate

    opt_rewards = np.array(opt_rewards)  # [N,]
    opt_actions = np.array(opt_actions)  # [N,]

    np.random.seed(seed)
    idx = np.arange(num_contexts)
    np.random.shuffle(idx)

    dataset = dataset[idx]
    opt_rewards = opt_rewards[idx]
    opt_actions = opt_actions[idx]
    opt_vals = (opt_rewards, opt_actions)  # ([N,], [N,])

    return dataset, opt_vals  # [N,7], ([N,], [N,])