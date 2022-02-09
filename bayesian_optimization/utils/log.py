import torch
import time
import logging
from collections import OrderedDict
import re
import matplotlib
from matplotlib import pyplot as plt
from os.path import split, splitext


def get_logger(filename, mode='a'):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    # 코드 실행 시 run 여러번 돌리면 logger에 handler가 중복되므로, 매번 삭제해줘야
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)
    logger.addHandler(logging.FileHandler(filename, mode=mode))
    logger.addHandler(logging.StreamHandler())
    return logger


class RunningAverage(object):
    def __init__(self, *keys):
        self.sum = OrderedDict()
        self.cnt = OrderedDict()
        self.clock = time.time()
        for key in keys:
            self.sum[key] = 0
            self.cnt[key] = 0

    def update(self, key, val):
        if isinstance(val, torch.Tensor):
            val = val.item()
        if self.sum.get(key, None) is None:
            self.sum[key] = val
            self.cnt[key] = 1
        else:
            self.sum[key] = self.sum[key] + val
            self.cnt[key] += 1

    def reset(self):
        for key in self.sum.keys():
            self.sum[key] = 0
            self.cnt[key] = 0
        self.clock = time.time()

    def clear(self):
        self.sum = OrderedDict()
        self.cnt = OrderedDict()
        self.clock = time.time()

    def keys(self):
        return self.sum.keys()

    def get(self, key):
        assert(self.sum.get(key, None) is not None)
        return self.sum[key] / self.cnt[key]

    def info(self, show_et=True):
        line = ''
        for key in self.sum.keys():
            val = self.sum[key] / self.cnt[key]
            if type(val) == float:
                line += f'{key} {val:.4f} '
            else:
                line += f'{key} {val} '.format(key, val)
        if show_et:
            line += f'({time.time()-self.clock:.3f} secs)'
        return line


def get_log(fileroot):
    step = []
    loss = []
    train_time = []
    eval_time = []
    ctxll = []
    tarll = []
    file = open(fileroot, "r")
    lines = file.readlines()
    for line in lines:
        # training step
        if "step" in line:
            linesplit = line.split(" ")
            step += [int(linesplit[3])]
            _loss = linesplit[-3]
            loss += [100 if _loss=="nan" else float(_loss)]
            train_time += [float(linesplit[-2][1:])]
        # evaluation step
        elif "ctx_ll" in line:
            linesplit = line.split(" ")
            ctxll += [float(linesplit[-5])]
            tarll += [float(linesplit[-3])]
            eval_time += [float(linesplit[-2][1:])]
    
    return step, loss, None, ctxll, tarll


def plot_log(fileroot, x_begin=None, x_end=None):
    step, loss, stepll, ctxll, tarll = get_log(fileroot)
    step = list(map(int, step))
    loss = list(map(float, loss))
    ctxll = list(map(float, ctxll))
    tarll = list(map(float, tarll))
    stepll = list(map(int, stepll)) if stepll else None
    
    if x_begin is None:
        x_begin = 0
    if x_end is None:
        x_end = step[-1]
    
    print_freq = 1 if len(step) == 1 else step[1] - step[0]

    plt.clf()
    plt.plot(step[x_begin//print_freq:x_end//print_freq],
             loss[x_begin//print_freq:x_end//print_freq])
    plt.xlabel('step')
    plt.ylabel('loss')

    directory, file = split(fileroot)
    filename = splitext(file)[0]
    plt.savefig(directory + "/" + filename + f"-{x_begin}-{x_end}.png")
    plt.clf()  # clear current figure
