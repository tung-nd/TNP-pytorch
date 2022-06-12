import yaml

from runner.args import get_args

args = get_args()

with open("paths.yaml") as f:
    paths = yaml.safe_load(f)
    datasets_path = paths["datasets_path"]
    evalsets_path = paths["evalsets_path"]
    results_path = paths["results_path"]
