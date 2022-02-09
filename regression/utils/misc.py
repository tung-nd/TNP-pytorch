import os
from importlib.machinery import SourceFileLoader
import math
import torch

def gen_load_func(parser, func):
    def load(args, cmdline):
        sub_args, cmdline = parser.parse_known_args(cmdline)
        for k, v in sub_args.__dict__.items():
            args.__dict__[k] = v
        return func(**sub_args.__dict__), cmdline
    return load


def load_module(filename):
    module_name = os.path.splitext(os.path.basename(filename))[0]
    return SourceFileLoader(module_name, filename).load_module()
    # <module "module_name" from "filename">
    #
    # ex.
    # <module "cnp" from "models/cnp.py">


def logmeanexp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def stack(x, num_samples=None, dim=0):
    return x if num_samples is None \
            else torch.stack([x]*num_samples, dim=dim)


def hrminsec(duration):
    hours, left = duration // 3600, duration % 3600
    mins, secs = left // 60, left % 60
    return f"{hours}hrs {mins}mins {secs}secs"