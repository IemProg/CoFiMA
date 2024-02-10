import json
import argparse
from trainer import train
from evaluator import test
import wandb

def main():
    args = setup_parser().parse_args()
    wt_alpha = args.wt_alpha

    seed = args.seed
    fisher_weighting = args.fisher_weighting
    wandb_mode = args.wandb_mode
    gpus = args.gpus

    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    # restore dataset name from arguments
    args["wt_alpha"] = wt_alpha

    args["fisher_weighting"] = fisher_weighting
    args["wandb_mode"] = wandb_mode

    args["seed"] = [seed]
    args["gpus"] = gpus

    exp_name = args["experiment_name"] + "_" + args["dataset"] + "-B" + str(args["init_cls"]) + "Inc" + str(
        args["increment"])
    wandb.init(
        # Set the project where this run will be logged
        project="CoFiMA",
        # Group experimnets
        group=args["exp_grp"],
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=exp_name,
        entity="imad-ma",
        mode=args["wandb_mode"],
        # Track hyperparameters and run metadata
        config=args)

    if args['test_only']:
        test(args)
    else:
        train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param

def split_comma_separated(string):
    return string.split(',')

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/cofima/cofima_cifar.json',
                        help='Json file of settings.')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--experiment_name', type=str, default="Reproduce",
                        help='Name of the experiment.')
    parser.add_argument('--exp_grp', type=str, default="InitialBaselines",
                        help='Name of the experiment group.')

    parser.add_argument('--wt_alpha', type=float, default=0.5,
                        help='Alpha for weight averaging')

    parser.add_argument('--seed', type=int, default=1993,
                        help='seed')

    parser.add_argument('--wandb_mode', type=str, default="offline",
                        choices=["online", "offline"],
                        help="wandb mode(default=offline)")

    parser.add_argument('--gpus', type=split_comma_separated,
                        default= ["0"], help='ids of gpus')

    return parser


if __name__ == '__main__':
    main()
