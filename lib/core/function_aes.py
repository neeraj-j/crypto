from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from apex import amp

logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          lr_scheduler, output_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, samples in enumerate(train_loader):
        # measure data loading time
        optimizer.zero_grad()
        #with torch.cuda.amp.autocast():
        data_time.update(time.time() - end)
        sample = samples["net_input"]["source"].cuda()
        aes_sample = samples["net_input"]["aes"].cuda()
        output = model(aes_sample)
        # Vae encoder
        if isinstance(output, list):
            recon, enc, mu, log_var = output
            if not sample.shape[1] == recon.shape[1]:
                print("sample Shape: {}; output shape: {}".format(sample.shape[1], output.shape[1]), flush=True)

            loss_dic = criterion(recon, sample, mu, log_var)
            loss = loss_dic['loss']
            # Todo: print other losses also
        else:  # normal encoder
            if not sample.shape[1] == output.shape[1]:
                print("sample Shape: {}; output shape: {}".format(sample.shape[1], output.shape[1]), flush=True)
            loss = criterion(output, sample)

        # compute gradient and do update step
        #loss.backward(torch.ones_like(loss))
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(loss)

        #scaler.scale(loss).backward()

        optimizer.step()
        lr_scheduler.step()
        # measure accuracy and record loss
        losses.update(loss.item(), sample.size(0))
        # mse of 1 sample

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'LR: {LR:.3} \t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=sample.size(0)/batch_time.val,
                      LR=lr_scheduler.get_last_lr()[0], 
                      data_time=data_time, loss=losses)
            logger.info(msg)


def validate(config, val_loader, val_dataset, model, criterion, output_dir, device=torch.device("cuda:0")):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mse = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, samples in enumerate(val_loader):
            # compute output
            sample = samples["net_input"]["source"].to(device)
            aes_sample = samples["net_input"]["aes"].cuda()
            output = model(aes_sample)
            # Vae encoder
            if isinstance(output, list):
                recon, enc, mu, log_var = output
                if not sample.shape[1] == recon.shape[1]:
                    print("sample Shape: {}; output shape: {}".format(sample.shape[1], output.shape[1]), flush=True)

                loss_dic = criterion(recon, sample, mu, log_var)
                loss = loss_dic['loss']
                # Todo: print other losses also
            else:  # normal encoder
                loss = criterion(output, sample)

            num_samples = sample.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_samples)
            mse.update(output[0].square().mean().item(), 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'MSE: {mse.avg:.3f} \t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          mse=mse, loss=losses)
                logger.info(msg)

    return losses.avg


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


# quant aware training
def train_quant(config, train_loader, model, criterion, optimizer, epoch,
          lr_scheduler, output_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()
    device = next(model.parameters()).device

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        input = input.to(device)
        data_time.update(time.time() - end)
        # compute output
        output = model(input)

        target = target.to(device, non_blocking=True)
        target_weight = target_weight.to(device, non_blocking=True)
        #output = outputs
        loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        # For onecycle scheduler
        lr_scheduler.step()
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'LR: {LR:.3} \t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      LR=lr_scheduler.get_last_lr()[0],
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)
            if i == 100:
                break

