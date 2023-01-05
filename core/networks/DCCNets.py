# -*- coding: utf-8 -*-
"""
Networks for Deep Combined Computing.

@author: Xinzhe Luo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from UNet import ResidualEncoder, ResidualDecoder
from layers import ResBlock, AbstractFusion
from einops import rearrange
from core.register.SpatialTransformer import SpatialTransformer, ResizeTransform


class DCCNet(nn.Module):

    def __init__(self, in_channels, dimension=2, num_classes=4, init_features=16, num_blocks=4, norm_type='batch',
                 dropout=0.2, activation=nn.LeakyReLU(), switch_bn=True, **kwargs):
        super(DCCNet, self).__init__()
        self.in_channels = in_channels
        self.dimension = dimension
        self.num_classes = num_classes
        self.init_features = init_features
        self.num_blocks = num_blocks
        self.norm_type = norm_type
        self.dropout = dropout
        self.activation = activation
        self.switch_bn = switch_bn
        self.kwargs = kwargs
        self.modalities = self.kwargs.pop('modalities', None)
        self.tracking_running_stats = self.kwargs.pop('tracking_running_stats', True)
        self.momentum = self.kwargs.pop('momentum', 0.01)
        if self.switch_bn:
            assert self.modalities is not None
        self.compose_ddf = self.kwargs.pop('compose_ddf', False)
        self.use_atlas = self.kwargs.pop('use_atlas', False)

        self.Conv = nn.Conv2d if self.dimension == 2 else nn.Conv3d

        self.encoder = ResidualEncoder(in_channels, dimension, init_features, num_blocks, norm_type,
                                       dropout, activation, switch_bn,
                                       modalities=self.modalities,
                                       tracking_running_stats=self.tracking_running_stats,
                                       momentum=self.momentum)

        self.seg_decoder = ResidualDecoder(dimension, init_features, num_blocks, norm_type,
                                           dropout, activation, switch_bn=False,
                                           tracking_running_stats=self.tracking_running_stats,
                                           momentum=self.momentum)

        self.reg_decoder = ResidualDecoder(dimension, init_features, num_blocks, norm_type,
                                           dropout, activation, switch_bn=False,
                                           tracking_running_stats=self.tracking_running_stats,
                                           momentum=self.momentum)

        self.fuse = AbstractFusion()
        self.reg_trans = nn.ModuleDict()
        for i in range(num_blocks + 1):
            self.reg_trans['trans_%s' % i] = ResBlock(in_channels=2 ** i * self.init_features * 2,
                                                      num_features=2 ** i * self.init_features,
                                                      dimension=self.dimension,
                                                      norm_type=self.norm_type,
                                                      dropout=self.dropout,
                                                      activation=self.activation,
                                                      switch_bn=False,
                                                      track_running_stats=self.tracking_running_stats,
                                                      momentum=self.momentum)

        self.seg_output_conv = self.Conv(self.init_features, self.num_classes, 1)
        self.reg_output_convs = nn.ModuleDict()
        for i in range(self.num_blocks):
            self.reg_output_convs['output_conv_%s' % i] = \
                self.Conv(2 ** i * self.init_features,
                          self.dimension * (len(self.modalities) + 1) if self.use_atlas
                          else self.dimension * len(self.modalities), 1)
            nn.init.constant_(self.reg_output_convs['output_conv_%s' % i].weight, 0)
            nn.init.constant_(self.reg_output_convs['output_conv_%s' % i].bias, 0)

        self.resize = ResizeTransform(dimension)

    def forward(self, x):
        x, x_enc = self.encoder(x)

        seg_dec = self.seg_decoder(x=rearrange(x, 'B M ... -> (B M) ...'),
                                   x_enc=[rearrange(y, 'B M ... -> (B M) ...') for y in x_enc])
        seg = rearrange(self.seg_output_conv(seg_dec[0]), '(B M) ... -> B M ...', M=len(self.modalities))

        x_trans = self.reg_trans['trans_%s' % self.num_blocks](self.fuse(x))
        x_enc_trans = [self.reg_trans['trans_%s' % i](self.fuse(x_enc[i])) for i in range(self.num_blocks)]

        x_dec = self.reg_decoder(x_trans, x_enc_trans)

        flows = []
        for i in range(self.num_blocks):
            flows.append(self.resize(self.reg_output_convs['output_conv_%s' % i](x_dec[i]), factor=2 ** i))

        if self.compose_ddf:
            y = (SpatialTransformer(size=x.shape[3:], padding_mode='zeros').getComposedFlows(flows[::-1]))
        else:
            y = (torch.stack(flows).sum(dim=0))
        y = rearrange(y, 'B (M d) ... -> B M d ...', d=self.dimension)

        return y, torch.softmax(seg, dim=2)

    def seg(self, x):
        x, x_enc = self.encoder(x)

        seg_dec = self.seg_decoder(x=rearrange(x, 'B M ... -> (B M) ...'),
                                   x_enc=[rearrange(y, 'B M ... -> (B M) ...') for y in x_enc])
        seg = rearrange(self.seg_output_conv(seg_dec[0]), '(B M) ... -> B M ...', M=len(self.modalities))

        return torch.softmax(seg, dim=2)

    def reg(self, x):
        x, x_enc = self.encoder(x)

        x_trans = self.reg_trans['trans_%s' % self.num_blocks](self.fuse(x))
        x_enc_trans = [self.reg_trans['trans_%s' % i](self.fuse(x_enc[i])) for i in range(self.num_blocks)]

        x_dec = self.reg_decoder(x_trans, x_enc_trans)

        flows = []
        for i in range(self.num_blocks):
            flows.append(self.resize(self.reg_output_convs['output_conv_%s' % i](x_dec[i]), factor=2 ** i))

        if self.compose_ddf:
            y = (SpatialTransformer(size=x.shape[3:], padding_mode='zeros').getComposedFlows(flows[::-1]))
        else:
            y = (torch.stack(flows).sum(dim=0))

        y = rearrange(y, 'B (M d) ... -> B M d ...', d=self.dimension)
        return y
