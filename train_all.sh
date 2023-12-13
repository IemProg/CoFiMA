##!/bin/bash


CUDA_VISIBLE_DEVICES=0 python3 main.py --config=exps/slca/slca_cifar.json \
                           --experiment_name WA_SEED1993  --gpus 0 --epochs 20 \
                           --exp_grp LongerT_FisherWA_MainResults \
                           --convnet_type vit-b-p16 --init_w -1 --wt_alpha 0.5 --seed 1993

CUDA_VISIBLE_DEVICES=0 python3 main.py --config=exps/slca/slca_imgnetr.json \
                           --experiment_name WA_SEED1993  --gpus 0 --epochs 20 \
                           --exp_grp LongerT_FisherWA_MainResults \
                           --convnet_type vit-b-p16 --init_w -1 --wt_alpha 0.5 --seed 1993