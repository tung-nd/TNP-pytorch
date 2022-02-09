import os
import torch
import yaml

from runner.args import get_args


args = get_args()

os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu}"
device = torch.device('cuda' if ((args.gpu >= 0) & torch.cuda.is_available()) else 'cpu')
args.device = device

with open("paths.yaml") as f:
    paths = yaml.safe_load(f)
    datasets_path = paths["datasets_path"]
    evalsets_path = paths["evalsets_path"]
    results_path = paths["results_path"]
