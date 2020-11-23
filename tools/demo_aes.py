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
from utils import aes

import torch.nn.functional as F
import models
from glob import glob
import soundfile as sf
import sounddevice as sd
from time import time
import copy


def parse_args():
    old_sys_argv = sys.argv
    sys.argv = [old_sys_argv[0]] + ['--cfg=../experiments/aes_L5_c512.yaml']   # 0.0168
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args = parser.parse_args()

    return args

def toInt(feats):
    # normalize from 0-255
    feats = (feats - feats.min()) * (255 / (feats.max() - feats.min()))
    feats = feats.round().astype(int)
    return feats

def toFloat(aes_sources):
    aes_sources = np.array(aes_sources) / 255.0 - 0.5
    return aes_sources


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'demo')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # data files
    #datapath = "../../data/LibriSpeech/dev-clean/**/*.flac"
    datapath = "../../data/LibriSpeech/aes/**/*.flac"
    filelist = glob(datapath, recursive=True)
    #filelist = ["../../data/custom/rani_16.flac"]

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
    key = [143, 194, 34, 208, 145, 203, 230, 143, 177, 246, 97, 206, 145, 92, 255, 84]
    iv = [103, 35, 148, 239, 76, 213, 47, 118, 255, 222, 123, 176, 106, 134, 98, 92]
    moo = aes.AESModeOfOperation()

    for fname in filelist:
        if "aes" in os.path.basename(fname):
            continue
        fullwav, curr_sample_rate = sf.read(fname)
        fullenc, enc_sample_rate = sf.read(fname.replace(".flac", "-aes.flac"))
        target_size = fullwav.shape[0]
        # 128 bit multiple for aes
        target_size = target_size - (target_size % 16)
        wav = fullwav[:target_size]
        enc_float = fullenc[:target_size]
        #wav_int = toInt(wav)
        # Encrypt
        #_, _, enc = moo.encrypt(wav_int, moo.modeOfOperation["CBC"],
        #                                key, moo.aes.keySize["SIZE_128"], iv)
        #enc_float = toFloat(enc)
        enc_float = torch.from_numpy(enc_float).float()
        enc_float = postprocess(enc_float)
        enc_float = enc_float.unsqueeze(0)
        with torch.no_grad():
            start = time()
            out,_ = model(enc_float)
            print("time: {}".format(time() - start))
            out = out.squeeze(0)
            enc_float = enc_float.squeeze(0)
            print("Loss : {}".format(F.mse_loss(out, torch.from_numpy(wav))))

        #np.savetxt("../log/randsum_run1.csv", enc)
        sd.play(wav,curr_sample_rate)
        status = sd.wait()
        sd.play(out.numpy()*10,curr_sample_rate)
        status = sd.wait()
        sd.play(enc_float.numpy(),curr_sample_rate)
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
