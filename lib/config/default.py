
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.SEED = 1

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# model params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'wav2vec_1'
_C.MODEL.ENCODER = "[(1,2,3)]" 
_C.MODEL.DECODER = "[(1,2,3)]" 
_C.MODEL.DECODER = ''
_C.MODEL.BLOCK = 'BASIC'
_C.MODEL.GROUPED = False 
_C.MODEL.LOG_COMPRESSION = False 
_C.MODEL.HIDDEN_DIMS = 0   # output dim of encoder
_C.MODEL.LATENT_DIMS = 0  # same as max_sample_size
_C.MODEL.LOSS = False  # same as max_sample_size

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = ''
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.VALID_SET = 'valid'
_C.DATASET.DATA_FORMAT = 'flac'
_C.DATASET.SAMPLE_RATE = 16000 
_C.DATASET.NORMALIZE = True 
_C.DATASET.SKIP_INVALID_SIZE_INPUTS_VALID_TEST = False
_C.DATASET.MAX_TOKENS = 0
_C.DATASET.MAX_SAMPLE_SIZE = 0
_C.DATASET.MIN_SAMPLE_SIZE = 0
_C.DATASET.BATCH_SIZE = 0

# train
_C.TRAIN = CN()

_C.TRAIN.LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.SCHEDULER = ''

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.LR = 0.0
_C.TRAIN.MIN_LR = 0.0
_C.TRAIN.MAX_LR = 0.0
_C.TRAIN.LAMBDA_DIAG = 10
_C.TRAIN.LAMBDA_OFFSET = 5

_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 8

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    cfg.DATASET.ROOT = os.path.join(
        cfg.DATA_DIR, cfg.DATASET.ROOT
    )


    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

