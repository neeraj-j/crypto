# ---------------------------------------------
# Test cases:
# Test case 1: play org, enc and dec data
# test case 2: plot all three in histogram
# test case3: if same file gets encoded differently
# ---------------------------------------------
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


def parse_args():
    old_sys_argv = sys.argv
    #sys.argv = [old_sys_argv[0]] + ['--cfg=../experiments/cbr_L4_grp.yaml']   # 0.002
    sys.argv = [old_sys_argv[0]] + ['--cfg=../experiments/cbr_L5_c64_150k.yaml']   # 0.0168
    #sys.argv = [old_sys_argv[0]] + ['--cfg=../experiments/cbr_L5_c64_lin.yaml']
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
    #filelist = glob(datapath, recursive=True)
    filelist = ["../../data/custom/rani_16.flac"]

    model = eval('models.'+cfg.MODEL.NAME+'.get_wav2vec')(
        cfg, is_train=True
    )

    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth')

    if os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))
        #logger.info("=> loaded checkpoint '{}')".format( checkpoint_file))

    model.eval()

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
            out, enc = model(feats)
            print("time: {}".format(time() - start))
            print("Loss : {}".format(F.mse_loss(out, feats)))
            out = out.squeeze(0)
            enc = enc.squeeze(0)
            #enc = torch.flatten(enc, start_dim=0)
            enc = enc[0]

        sd.play(wav,curr_sample_rate)
        status = sd.wait()
        sd.play(out.numpy()/10,curr_sample_rate)
        status = sd.wait()
        sd.play(enc.numpy(),curr_sample_rate)
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
