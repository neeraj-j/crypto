 # based on https://github.com/AntixK/PyTorch-VAE/blob/master/models/dip_vae.py

import logging
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TypeVar
Tensor = TypeVar('torch.tensor')


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


enc_blocks = {"BASIC": ConvBase, "CBR": ConvBNReLU}
dec_blocks = {"BASIC": ConvTBase, "CBR": ConvTBNReLU}

class Wav2VecModel_Dip(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.lambda_diag = cfg.TRAIN.LAMBDA_DIAG    # 10
        self.lambda_offdiag = cfg.TRAIN.LAMBDA_OFFSET   # 5

        feature_enc_layers = eval(cfg.MODEL.ENCODER)
        grouped = cfg.MODEL.GROUPED
        self.encoder = Encoder(
            feature_enc_layers,
            cfg.MODEL.BLOCK,
            grouped,
            cfg.MODEL.HIDDEN_DIMS,
            cfg.MODEL.LATENT_DIMS,
        )

        # decoder

        feature_dec_layers = eval(cfg.MODEL.DECODER)
        self.decoder = Decoder(
            feature_dec_layers,
            cfg.MODEL.BLOCK,
            grouped,
            self.encoder.in_d,
            cfg.MODEL.HIDDEN_DIMS,
            cfg.MODEL.LATENT_DIMS,
        )

    def forward(self, source):

        source = source.unsqueeze(1)
        z, mu, log_var = self.encoder(source)

        x = self.decoder(z)
        x = x.squeeze(1)

        return [x, z, mu, log_var]

    # using dip loss
    def loss_function(self, recons, input, mu, log_var):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        kld_weight = 1 #* kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input, reduction='sum')


        kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        # DIP Loss
        centered_mu = mu - mu.mean(dim=1, keepdim = True) # [B x D]
        cov_mu = centered_mu.t().matmul(centered_mu).squeeze() # [D X D]

        # Add Variance for DIP Loss II
        cov_z = cov_mu + torch.mean(torch.diagonal((2. * log_var).exp(), dim1 = 0), dim = 0) # [D x D]
        # For DIp Loss I
        # cov_z = cov_mu

        cov_diag = torch.diag(cov_z) # [D]
        cov_offdiag = cov_z - torch.diag(cov_diag) # [D x D]
        dip_loss = self.lambda_offdiag * torch.sum(cov_offdiag ** 2) + \
                   self.lambda_diag * torch.sum((cov_diag - 1) ** 2)

        loss = recons_loss + kld_weight * kld_loss + dip_loss
        return {'loss': loss,
                'Reconstruction_Loss':recons_loss,
                'KLD':-kld_loss,
                'DIP_Loss':dip_loss}


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
        hidden_dim,
        latent_dim
    ):
        super().__init__()

        self.in_d = 1
        # Todo: convert it into nn.Sequnetial as done in dip_vae
        # once I fix dimention issue
        self.conv_layers = nn.ModuleList()
        block = enc_blocks[block_name] 
        for dim, k, stride in conv_layers:
            grp = self.in_d if grouped else 1
            self.conv_layers.append(block(self.in_d, dim, k, stride, groups=grp))
            self.in_d = dim

        self.fc_mu = nn.Linear(hidden_dim*self.in_d, latent_dim)
        self.fc_var = nn.Linear(hidden_dim*self.in_d, latent_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        # BxT -> BxCxT

        for conv in self.conv_layers:
            x = conv(x)

        x = torch.flatten(x, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        z = self.reparameterize(mu, log_var)

        return [z, mu, log_var]


class Decoder(nn.Module):
    def __init__(
        self,
        conv_layers,
        block_name,
        grouped,
        in_d,
        hidden_dim,
        latent_dim
    ):
        super().__init__()

        self.in_c = in_d
        self.hidden_d = hidden_dim
        self.decoder_in = nn.Linear(latent_dim,hidden_dim*in_d)
        self.conv_layers = nn.ModuleList()
        block = dec_blocks[block_name] 
        for dim, k, stride in conv_layers[:-1]:
            grp = in_d if grouped else 1
            self.conv_layers.append(block(in_d, dim, k, stride, groups=grp))
            in_d = dim

        # final layer with basic block
        block = dec_blocks['BASIC']
        dim, k, stride = conv_layers[-1]
        #grp = in_d if grouped else 1
        grp = 1
        self.conv_layers.append(block(in_d, 1, k, stride, groups=grp))


    def forward(self, x):
        x = self.decoder_in(x)
        x = x.view(-1, self.in_c, self.hidden_d)
        for conv in self.conv_layers:
            x = conv(x)

        return x


def get_wav2vec(cfg, is_train=False):
    model = Wav2VecModel_Dip(cfg)
    if is_train :
        model.init_weights()

    return model

