
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models
from apex import amp

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_wav2vec')(
        cfg, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    dump_input = torch.rand(
        (1, cfg.DATASET.MAX_SAMPLE_SIZE)
    )
    logger.info(get_model_summary(model, dump_input))

    model = model.cuda()

    # define loss function (criterion) and optimizer
    if cfg.MODEL.LOSS:
        criterion = model.loss_function
    else:
        criterion = nn.MSELoss(reduction = 'mean').cuda()

    manifest = os.path.join(cfg.DATASET.ROOT,"{}.tsv".format("train"))
    train_dataset = dataset.FileAudioDataset(
            manifest, 
            sample_rate=cfg.DATASET.SAMPLE_RATE,
            max_sample_size=cfg.DATASET.MAX_SAMPLE_SIZE,
            min_sample_size=cfg.DATASET.MAX_SAMPLE_SIZE,
            min_length=cfg.DATASET.MIN_SAMPLE_SIZE,
            pad=False,
            normalize=cfg.DATASET.NORMALIZE,
            )
    manifest = os.path.join(cfg.DATASET.ROOT,"{}.tsv".format("valid"))
    valid_dataset = dataset.FileAudioDataset(
            manifest, 
            sample_rate=cfg.DATASET.SAMPLE_RATE,
            max_sample_size=cfg.DATASET.MAX_SAMPLE_SIZE,
            min_sample_size=cfg.DATASET.MAX_SAMPLE_SIZE,
            min_length=cfg.DATASET.MIN_SAMPLE_SIZE,
            pad=False,
            normalize=cfg.DATASET.NORMALIZE,
            )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE*len(cfg.GPUS),
        collate_fn=train_dataset.collater,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE*len(cfg.GPUS),
        collate_fn=valid_dataset.collater,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    optimizer = get_optimizer(cfg, model)
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2')   #Neeraj: MP
    # using cosine
    steps_per_epoch = train_loader.__len__()
    print("Steps per epoch: {}".format(steps_per_epoch))
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.TRAIN.END_EPOCH, eta_min=cfg.TRAIN.MIN_LR, last_epoch=-1)
        # using OneCycleLR
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, cfg.TRAIN.LR, steps_per_epoch=int(steps_per_epoch),
        epochs=cfg.TRAIN.END_EPOCH,
        last_epoch=-1
    )


    best_perf = 0.0
    best_model = False
    last_epoch = -1
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth')

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        amp.load_state_dict(checkpoint['amp'])

        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch, lr_scheduler,
              final_output_dir)

        # evaluate on validation set
        perf_indicator = validate(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir)
        # cosinelr Step is done inside train, after every epoch
        #lr_scheduler.step()

        if perf_indicator <= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
            'amp': amp.state_dict(),   #Neeraj: Mixed precision
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    #torch.save(model.module.state_dict(), final_model_state_file)
    torch.save(model.state_dict(), final_model_state_file)


if __name__ == '__main__':
    main()
