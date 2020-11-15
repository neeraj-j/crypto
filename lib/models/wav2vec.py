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

class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
         output = F.group_norm(
                 input.float(),
                 self.num_groups,
                 self.weight.float() if self.weight is not None else None,
                 self.bias.float() if self.bias is not None else None,
                 self.eps,
                 )
         return output.type_as(input)

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


class ConvReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, bias=False, momentum=0.1, groups=1):
        super(ConvReLU, self).__init__(
            nn.Conv1d(in_planes, out_planes, kernel_size, stride, padding, bias=bias, groups=groups),
            nn.ReLU(inplace=True)
            )

class ConvGNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, bias=False, affine=True, groups=1):
        super(ConvGNReLU, self).__init__(
            nn.Conv1d(in_planes, out_planes, kernel_size, stride, padding, bias=bias, groups=groups),
            nn.GroupNorm(1, out_planes, affine=affine),
            nn.ReLU(inplace=True)
        )

class ConvTGNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, bias=False, affine=True, groups=1):
        super(ConvTGNReLU, self).__init__(
            nn.Conv1d(in_planes, out_planes, kernel_size, stride, padding, bias=bias, groups=groups),
            nn.GroupNorm(1, out_planes, affine=affine),
            nn.ReLU(inplace=True)
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


enc_blocks = {"BASIC": ConvBase, "CBR": ConvBNReLU, "CGR":ConvGNReLU}
dec_blocks = {"BASIC": ConvTBase, "CBR": ConvTBNReLU, "CGR":ConvTGNReLU}

class Wav2VecModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        feature_enc_layers = eval(cfg.MODEL.ENCODER)
        grouped = cfg.MODEL.GROUPED
        self.encoder = Encoder(
            feature_enc_layers,
            cfg.MODEL.BLOCK,
            grouped,
            log_compression=cfg.MODEL.LOG_COMPRESSION,
        )

        feature_dec_layers = eval(cfg.MODEL.DECODER)
        self.decoder = Decoder(
                feature_dec_layers[:-1],
            cfg.MODEL.BLOCK,
            grouped,
            self.encoder.in_d
        )

        block = dec_blocks['BASIC'] 
        for dim, k, stride in feature_dec_layers[-1:]:
            grp = self.decoder.in_d if grouped else 1
            self.final_layer =  block(self.decoder.in_d, 1, k, stride) 


    def forward(self, source):

        source = source.unsqueeze(1)
        features = self.encoder(source)

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
        grouped,
        log_compression
    ):
        super().__init__()

        self.in_d = 1
        self.conv_layers = nn.ModuleList()
        block = enc_blocks[block_name] 
        for dim, k, stride in conv_layers:
            grp = self.in_d if grouped else 1
            self.conv_layers.append(block(self.in_d, dim, k, stride, groups=grp))
            self.in_d = dim

        self.log_compression = log_compression

    def forward(self, x):
        # BxT -> BxCxT

        for conv in self.conv_layers:
            x = conv(x)

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

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
    model = Wav2VecModel(cfg)
    if is_train :
        model.init_weights()

    return model

