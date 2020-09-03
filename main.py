import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import random
import argparse
import numpy as np

import tensorflow as tf

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from model.detr import DETR

def build_detr(args, num_classes=91, num_queries=100):
    if args.backbone == 'resnet50':
        return DETR()


def get_argument():
    parser = argparse.ArgumentParser()

    # backbone
    parser.add_argument("--backbone",           type=str,       default='resnet50',
                        help="Name of the convolutional backbone to use")
    parser.add_argument("--dilation",           action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument("--position_embedding", type=str,       default='sine',     choices=['sine', 'learned'],    
                        help="Type of positional embedding to use on top of the image features")

    # transformer
    parser.add_argument("--enc_layers",         type=int,       default=6,
                        help="Number of encoding layers in the transformer")
    parser.add_argument("--dec_layers",         type=int,       default=6,
                        help="Number of decoding layers in the transformer")
    parser.add_argument("--dim_feedforward",    type=int,       default=2048,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument("--hidden_dim",         type=int,       default=256,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument("--dropout",            type=float,     default=0.1,
                        help="Dropout applied in the transformer")
    parser.add_argument("--nheads",             type=int,       default=8,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument("--num_queries",        type=int,       default=100,
                        help="Number of query slots")
    parser.add_argument("--pre_norm",           action='store_true')

    # loss
    parser.add_argument('--no_aux_loss',        action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # matcher
    parser.add_argument("--set_cost_class",     type=float,     default=1,
                        help="Class coefficient in the matching cost")
    parser.add_argument("--set_cost_bbox",      type=float,     default=5,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument("--set_cost_giou",      type=float,     default=2,
                        help="giou box coefficient in the matching cost")

    # loss coefficients
    parser.add_argument('--mask_loss_coef',     type=float,     default=1)
    parser.add_argument('--dice_loss_coef',     type=float,     default=1)
    parser.add_argument('--bbox_loss_coef',     type=float,     default=5)
    parser.add_argument('--giou_loss_coef',     type=float,     default=2)
    parser.add_argument('--eos_coef',           type=float,     default=0.1,
                        help="Relative classification weight of the no-object class")


    # hyperparameter
    parser.add_argument("--classes",            type=int,       default=20)
    parser.add_argument("--lr",                 type=float,     default=1e-4)
    parser.add_argument("--lr_backbone",        type=float,     default=1e-5)
    parser.add_argument("--batch_size",         type=int,       default=2)
    parser.add_argument("--weight_decay",       type=float,     default=1e-4)
    parser.add_argument("--epochs",             type=int,       default=300)
    

    # etc
    parser.add_argument("--summary",            action='store_true')
    parser.add_argument('--baseline-path',      type=str,       default='/workspace/src/Challenge/code_baseline')
    parser.add_argument('--src-path',           type=str,       default='.')
    parser.add_argument('--data-path',          type=str,       default=None)
    parser.add_argument('--result-path',        type=str,       default='./result')
    parser.add_argument('--snapshot',           type=str,       default=None)
    parser.add_argument("--gpus",               type=str,       default=-1)
    parser.add_argument("--ignore-search",      type=str,       default='')

    return parser.parse_args()

def main():
    args = get_argument()

    sys.path.append(args.baseline_path)
    from common import get_logger
    from common import get_session

    logger = get_logger("MyLogger")

    args, initial_epoch = search_same(args)
    if initial_epoch == -1:
        # training was already finished!
        return

    get_session(args)
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))



if __name__ == '__main__':
    main()