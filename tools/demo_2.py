# ---------------------------------------------
# Test cases:
# Test case 1: play org, enc and dec data
# test case 2: plot all three in histogram
# test case3: if same file gets encoded differently
# ---------------------------------------------
# This program run1 and run2 and try to cross decode
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os, sys
import pprint
import shutil
import numpy as np

import torch
import torch.nn.functional as F

import _init_paths
from config import cfg
from config import update_config
from utils.utils import create_logger

import torch.nn.functional as F
import models
from glob import glob
import soundfile as sf
import sounddevice as sd
from time import time
import copy


def parse_args():
    old_sys_argv = sys.argv
    #sys.argv = [old_sys_argv[0]] + ['--cfg=../experiments/cbr_L5_c64_150k.yaml']   # 0.0168
    sys.argv = [old_sys_argv[0]] + ['--cfg=../experiments/cbr_L5_c128_vq.yaml']   # 0.0168
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'demo')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # data files
    datapath = "../../data/LibriSpeech/dev-clean/**/*.flac"
    filelist = glob(datapath, recursive=True)
    #filelist = ["../../data/custom/rani_16.flac"]

    model = eval('models.'+cfg.MODEL.NAME+'.get_wav2vec')(
        cfg, is_train=True
    )

    model2 = copy.deepcopy(model)

    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint_run1.pth')

    if os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

        # lead second run checkpoint
        checkpoint_file = checkpoint_file.replace("run1","run2")
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        model2.load_state_dict(checkpoint['state_dict'])

        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))
        #logger.info("=> loaded checkpoint '{}')".format( checkpoint_file))

    model.eval()
    model2.eval()
    VQ = True
    for fname in filelist:
        fullwav, curr_sample_rate = sf.read(fname)
        target_size = fullwav.shape[0]
        #target_size = 20000
        target_size = target_size - (target_size % 40)
        wav = fullwav[:target_size]
        feats = torch.from_numpy(wav).float()
        feats = postprocess(feats)
        feats = feats.unsqueeze(0)
        with torch.no_grad():
            start = time()
            out = model(feats)
            feats = feats.unsqueeze(1)
            enc1 = model.encoder(feats)
            dec = model.decoder(enc1)
            if VQ:
                out1 = dec
            else:
                out1 = model.final_layer(dec)
            out1 = out1.squeeze(1)
            enc2 = model2.encoder(feats)
            dec = model2.decoder(enc1)
            if VQ:
                out2 = dec
            else:
                out2 = model2.final_layer(dec)
            out2 = out2.squeeze(1)
            print("time: {}".format(time() - start))
            print("Loss1 : {}".format(F.mse_loss(out1, feats)))
            print("Loss2 : {}".format(F.mse_loss(out2, feats)))
            out1 = out1.squeeze(0)
            out2 = out2.squeeze(0)

        sd.play(wav,curr_sample_rate)
        status = sd.wait()
        sd.play(out1.numpy()/10,curr_sample_rate)
        status = sd.wait()
        sd.play(out2.numpy()/10,curr_sample_rate)
        status = sd.wait()
        continue



def postprocess( feats):
    if feats.dim() == 2:
        feats = feats.mean(-1)

    assert feats.dim() == 1, feats.dim()

    with torch.no_grad():
         feats = F.layer_norm(feats, feats.shape)
    return feats


if __name__ == '__main__':
    main()
