import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--expid', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda') # 'cpu' to use cpu

    # wheel
    parser.add_argument("--cmab_data", choices=["wheel"], default="wheel")
    parser.add_argument("--cmab_wheel_delta", type=float, default=0.5)
    parser.add_argument("--cmab_mode", choices=["train", "eval", "plot", "evalplot"], default="train")
    parser.add_argument('--cmab_num_bs', type=int, default=10)
    parser.add_argument("--cmab_train_update_freq", type=int, default=1)
    parser.add_argument("--cmab_train_num_batches", type=int, default=1)
    parser.add_argument("--cmab_train_batch_size", type=int, default=8)
    parser.add_argument("--cmab_train_seed", type=int, default=0)
    parser.add_argument("--cmab_train_reward", type=str, default="all")
    parser.add_argument("--cmab_eval_method", type=str, default="ucb")
    parser.add_argument("--cmab_eval_num_contexts", type=int, default=2000)
    parser.add_argument("--cmab_eval_seed_start", type=int, default=0)
    parser.add_argument("--cmab_eval_seed_end", type=int, default=49)
    parser.add_argument("--cmab_plot_seed_start", type=int, default=0)
    parser.add_argument("--cmab_plot_seed_end", type=int, default=49)

    # Model
    parser.add_argument('--model', type=str, default="tnpa")

    # Training
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--save_freq', type=int, default=1000)

    args = parser.parse_args()

    return args