# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, bias=False, momentum=0.1, groups=1):
        super(ConvBNReLU, self).__init__(
            nn.Conv1d(in_planes, out_planes, kernel_size, stride, padding, bias=bias, groups=groups),
            nn.BatchNorm1d(out_planes, momentum=momentum),
            nn.ReLU(inplace=True)
        )

class ConvTBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, bias=False, momentum=0.1, groups=1):
        super(ConvTBNReLU, self).__init__(
            nn.ConvTranspose1d(in_planes, out_planes, kernel_size, stride, padding, bias=bias, groups=groups),
            nn.BatchNorm1d(out_planes, momentum=momentum),
            nn.ReLU(inplace=True)
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, bias=False, momentum=0.1, groups=1):
        super(ConvBN, self).__init__(
            nn.Conv1d(in_planes, out_planes, kernel_size, stride, padding, bias=bias, groups=groups),
            nn.BatchNorm1d(out_planes, momentum=momentum),
        )


class ConvTBase(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, bias=False, momentum=0.1, groups=1):
        super(ConvTBase, self).__init__(
            nn.ConvTranspose1d(in_planes, out_planes, kernel_size, stride, padding, bias=bias, groups=groups),
        )

class ConvBase(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, bias=False, momentum=0.1, groups=1):
        super(ConvBase, self).__init__(
            nn.Conv1d(in_planes, out_planes, kernel_size, stride, padding, bias=bias, groups=groups),
        )


class Bottleneck(nn.Module):
    expansion: int = 1

    def __init__( self, inplanes, planes, kernel_size, stride, padding=0, bias=False):
        super(Bottleneck, self).__init__()
        width = int(planes / self.expansion) 
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ConvBNReLU(inplanes, width, 1,1)
        self.conv2 = ConvBNReLU(width, width, kernel_size,stride, padding)
        self.conv3 = ConvBN(width, planes, 1,1) #convBN if relu
        self.relu = nn.ReLU(inplace=True)
        if stride !=1 or inplanes !=planes*self.expansion:
            self.downsample = ConvBN(inplanes, planes, 1,stride)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x)
        out = self.conv3(out)
        identity = self.downsample(x)
        out += identity[:,:,:out.shape[2]]
        out = self.relu(out)
        return out


enc_blocks = {"BASIC": ConvBase, "CBR": ConvBNReLU, "BOT":Bottleneck }
dec_blocks = {"BASIC": ConvTBase, "CBR": ConvTBNReLU,}

class Res2WavModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        feature_enc_layers = eval(cfg.MODEL.ENCODER)
        grouped = cfg.MODEL.GROUPED
        dim, k, stride = feature_enc_layers[0]
        self.first_layer = ConvBNReLU(1,dim, k,stride) 
        self.encoder = Encoder(
            feature_enc_layers[1:],
            cfg.MODEL.BLOCK,
            dim,
        )

        feature_dec_layers = eval(cfg.MODEL.DECODER)
        self.decoder = Decoder(
                feature_dec_layers[:-1],
            "CBR",
            grouped,
            self.encoder.in_d
        )

        block = dec_blocks['BASIC'] 
        for dim, k, stride in feature_dec_layers[-1:]:
            grp = self.decoder.in_d if grouped else 1
            self.final_layer =  block(self.decoder.in_d, 1, k, stride) 


    def forward(self, source):

        source = source.unsqueeze(1)
        features = self.first_layer(source)
        features = self.encoder(features)

        x = self.decoder(features)
        x = self.final_layer(x)
        x = x.squeeze(1)

        return x


    def init_weights(self, pretrained='', verbose=True):
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.normal_(m.weight, std=0.001)
                    for name, _ in m.named_parameters():
                        if name in ['bias']:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose1d):
                    nn.init.normal_(m.weight, std=0.001)
                    for name, _ in m.named_parameters():
                        if name in ['bias']:
                            nn.init.constant_(m.bias, 0)



class Encoder(nn.Module):
    def __init__(
        self,
        conv_layers,
        block_name,
        in_d,
    ):
        super().__init__()

        self.in_d = in_d
        self.conv_layers = nn.ModuleList()
        block = enc_blocks[block_name] 
        for dim, k, stride in conv_layers:
            pad = 1 if stride == 1 else 0
            self.conv_layers.append(block(self.in_d, dim, k, stride, pad))
            self.in_d = dim

    def forward(self, x):
        # BxT -> BxCxT

        for conv in self.conv_layers:
            x = conv(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        conv_layers,
        block_name,
        grouped,
        in_d
    ):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        block = dec_blocks[block_name] 
        self.in_d = in_d
        for dim, k, stride in conv_layers:
            grp = self.in_d if grouped else 1
            self.conv_layers.append(block(self.in_d, dim, k, stride, groups=grp))
            self.in_d = dim


    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)

        return x


def get_wav2vec(cfg, is_train=False):
    model = Res2WavModel(cfg)
    #if is_train :
    #    model.init_weights()

    return model

