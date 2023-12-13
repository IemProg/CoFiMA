import json
import argparse
from trainer import train
from evaluator import test
import wandb

def main():
    args = setup_parser().parse_args()

    epochs = args.epochs
    adapter_bottleneck = args.adapter_bottleneck
    convnet_type     = args.convnet_type
    ffn_adapter_scalar = args.ffn_adapter_scalar
    ffn_option = args.ffn_option
    adapter_init = args.adapter_init
    adapter_residual = args.adapter_residual

    disable_ca = args.disable_ca
    ca_epochs = args.ca_epochs
    wt_alpha = args.wt_alpha
    init_w = args.init_w
    dist_estim = args.dist_estim

    seed = args.seed
    fisher_weighting = args.fisher_weighting
    ensembling = args.ensembling
    ensembling_init = args.ensembling_init

    wise_ft = args.wise_ft
    wandb_mode = args.wandb_mode
    gpus = args.gpus
    ema  = args.ema
    ema_beta = args.ema_beta
    ema_update = args.ema_update

    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    # restore dataset name from arguments
    args["epochs"] = int(epochs)
    args["adapter_bottleneck"] = adapter_bottleneck
    args["convnet_type"] = convnet_type
    args["ffn_adapter_scalar"] = ffn_adapter_scalar
    args["ffn_option"] = ffn_option
    args["adapter_init"] = adapter_init
    args["adapter_residual"] = adapter_residual

    args["disable_ca"] = disable_ca

    args["ca_epochs"] = ca_epochs
    args["wt_alpha"] = wt_alpha
    args["init_w"] = init_w
    args["dist_estim"] = dist_estim

    args["fisher_weighting"] = fisher_weighting
    args["ensembling"] = ensembling
    args["ensembling_init"] = ensembling_init

    args["wise_ft"] = wise_ft
    args["wandb_mode"] = wandb_mode

    args["seed"] = [seed]
    args["gpus"] = gpus
    args["ema"] = ema
    args["ema_beta"] = ema_beta
    args["ema_update"] = ema_update

    exp_name = args["experiment_name"] + "_" + args["dataset"] + "-B" + str(args["init_cls"]) + "Inc" + str(
        args["increment"])
    wandb.init(
        # Set the project where this run will be logged
        project="GOP",
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
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--experiment_name', type=str, default="GOP CIFAR100",
                        help='Name of the experiment.')
    parser.add_argument('--exp_grp', type=str, default="InitialBaselines",
                        help='Name of the experiment group.')
    parser.add_argument('--convnet_type', type=str, default="vit-b-p16-adapters",
                        help='Tyoe of convnet to use')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs')
    parser.add_argument('--adapter_bottleneck', type=int, default=64,
                        help='Size of adapter bottleneck')
    # Add the argument
    parser.add_argument('--thresh', type=float, nargs='+',
                        default=[0.90], help='A list of threshold values for GOP')
    parser.add_argument('--mixup_weight', type=float, default=0.1,
                        help='mixup_weight for flatness minima')
    parser.add_argument('--mixup_alpha', type=int, default=20,
                        help='mixup_alpha for data-perturbation')

    parser.add_argument('--ffn_adapter_scalar', type=float, default=0.1,
                        help='Adapter scaling factor')
    parser.add_argument('--ffn_option', type=str, default="parallel",
                        choices=["sequential", "parallel"],
                        help='Type of convnet to use')
    parser.add_argument('--adapter_init', type=str, default="he_init",
                        choices=["he_init", "lora"],
                        help='Type of convnet to use')

    parser.add_argument('--adapter_residual', type=bool, default=False,
                        help='Residual adapter')

    parser.add_argument('--disable_ca', type=bool, default=False,
                        help='Enable class alignement')

    parser.add_argument('--ca_epochs', type=int, default=5,
                        help='Number of class-alignement epochs')

    parser.add_argument('--wt_alpha', type=float, default=1.0,
                        help='Alpha for weight averaging')
    parser.add_argument('--init_w', type=int, default=-1,
                        help='Interpolate with the init model having the index init_w')
    parser.add_argument('--dist_estim', type=str, default="gaussian",
                        choices=["gaussian", "kde", "gmm", "gmm_light"],
                        help='Distribution estimation method for feature vectors')

    parser.add_argument('--seed', type=int, default=1993,
                        help='seed')

    parser.add_argument('--fisher_weighting', action='store_true',
                        help="Use Fisher weighting for the model weights (default=False)")

    parser.add_argument('--ensembling', action='store_true',
                        help="Use weight-space ensembling for the models (default=False)")

    parser.add_argument('--ensembling_init', action='store_true',
                        help="Seperate Ensemble Averaging instead of recursive averaging (default=False)")

    parser.add_argument('--wise_ft', action='store_true',
                        help="Use wise_ft (default=False)")

    parser.add_argument('--wandb_mode', type=str, default="online",
                        choices=["online", "offline"],
                        help="wandb mode(default=Online)")

    parser.add_argument('--ema', action='store_true',
                        help="Use Exponential Moving Average (default=False)")
    parser.add_argument('--ema_beta', type=float, default=0.999,
                        help='Beta for Exponential Moving Average (default=0.999)')
    parser.add_argument('--ema_update', type=int, default=5,
                        help='Nbr of epochs for Exponential Moving Average update (default=5)')


    parser.add_argument('--dset_variant', type=int, default=0,
                        choices=[0, 1, 2, 3, 4],
                        help='seed')

    parser.add_argument('--gpus', type=split_comma_separated,
                        default= ["0"], help='ids of gpus')

    return parser


if __name__ == '__main__':
    main()
