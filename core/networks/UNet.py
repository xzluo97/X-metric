# -*- coding: utf-8 -*-
"""
Modules for network construction.

@author: Xinzhe Luo
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import layers


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=3, dimension=3, init_features=16, num_blocks=4, norm_type=None, dropout=0,
                 activation=nn.LeakyReLU(), **kwargs):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimension = dimension
        self.init_features = init_features
        self.num_blocks = num_blocks
        self.norm_type = norm_type
        self.dropout = dropout
        self.activation = activation
        self.kwargs = kwargs
        self.track_running_stats = self.kwargs.pop('track_running_stats', True)
        self.momentum = self.kwargs.pop('momentum', 0.01)
        self.output_levels = self.kwargs.pop('output_levels', (0,))
        if self.output_levels == (-1, ):
            self.output_levels = list(range(self.num_blocks))

        self.down_blocks = nn.ModuleDict()
        for i in range(self.num_blocks):
            down_in_channels = self.in_channels if i == 0 else 2 ** (i - 1) * self.init_features
            self.down_blocks['down_block_%s' % i] = layers.ConvBlock(down_in_channels,
                                                                     dimension=self.dimension,
                                                                     num_features=2 ** i * self.init_features,
                                                                     norm_type=self.norm_type, dropout=self.dropout,
                                                                     activation=self.activation,
                                                                     track_running_stats=self.track_running_stats,
                                                                     momentum=self.momentum)

        self.bottleneck_block = layers.ConvBlock(2 ** (self.num_blocks - 1) * self.init_features,
                                                 dimension=self.dimension,
                                                 num_features=2 ** self.num_blocks * self.init_features,
                                                 norm_type=self.norm_type, dropout=self.dropout,
                                                 activation=self.activation,
                                                 track_running_stats=self.track_running_stats,
                                                 momentum=self.momentum)

        self.upsamples = nn.ModuleDict()
        for i in range(self.num_blocks):
            self.upsamples['upsample_%s' % i] = layers.DeconvBlock(2 ** (i + 1) * self.init_features,
                                                                   out_channels=2 ** i * self.init_features,
                                                                   dimension=self.dimension,
                                                                   norm_type=self.norm_type,
                                                                   dropout=self.dropout,
                                                                   activation=self.activation,
                                                                   track_running_stats=self.track_running_stats,
                                                                   momentum=self.momentum)

        self.up_blocks = nn.ModuleDict()
        for i in range(self.num_blocks):
            self.up_blocks['up_block_%s' % i] = layers.ConvBlock(2 ** (i + 1) * self.init_features,
                                                                 dimension=self.dimension,
                                                                 num_features=2 ** i * self.init_features,
                                                                 norm_type=self.norm_type, dropout=self.dropout,
                                                                 activation=self.activation,
                                                                 track_running_stats=self.track_running_stats,
                                                                 momentum=self.momentum)

        if self.dimension == 2:
            self.max_pool = nn.MaxPool2d(2)
            self.output_convs = nn.ModuleDict()
            for i in self.output_levels:
                self.output_convs['output_conv_%s' % i] = nn.Conv2d(2 ** i * self.init_features,
                                                                    self.out_channels, 3, 1, 1)

        elif self.dimension == 3:
            self.max_pool = nn.MaxPool3d(2)
            self.output_convs = nn.ModuleDict()
            for i in self.output_levels:
                self.output_convs['output_conv_%s' % i] = nn.Conv3d(2 ** i * self.init_features,
                                                                    self.out_channels, 3, 1, 1)

        else:
            raise NotImplementedError

    def forward(self, x):
        x_enc = []
        for i in range(self.num_blocks):
            x_enc.append(self.down_blocks['down_block_%s' % i](x))
            x = self.max_pool(x_enc[-1])

        x = self.bottleneck_block(x)

        x_dec = [x]
        for i in range(self.num_blocks - 1, -1, -1):
            x = self.upsamples['upsample_%s' % i](x_dec[-1])
            x = torch.cat([x, x_enc[i]], dim=1)
            x_dec.append(self.up_blocks['up_block_%s' % i](x))

        x_dec = x_dec[::-1]

        if tuple(self.output_levels) == (0, ):
            y = self.output_convs['output_conv_0'](x_dec[0])
        else:
            y = dict([(i, self.output_convs['output_conv_%s' % i](x_dec[i])) for i in self.output_levels])

        return y


class SimpleUNet(nn.Module):
    def __init__(self, in_channels, out_channels=3, init_features=32, num_blocks=4, dropout=0,
                 activation=nn.LeakyReLU(), **kwargs):
        super(SimpleUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.kwargs = kwargs

        self.down_blocks = nn.ModuleDict()
        for i in range(num_blocks):
            down_in_channels = self.in_channels if i == 0 else 2**(i-1)*init_features
            self.down_blocks['down_block_%s' % i] = layers.ConvBlock(down_in_channels, num_layers=1,
                                                                     num_features=2**i*init_features, stride=2,
                                                                     padding=1, batch_norm=False, dropout=dropout,
                                                                     activation=activation)

        self.bottleneck_block = layers.ConvBlock(2 ** (num_blocks - 1) * init_features, 1,
                                                 2 ** (num_blocks-2) * init_features, batch_norm=False,
                                                 dropout=dropout, activation=activation)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear')
        self.up_blocks = nn.ModuleDict()
        for i in range(num_blocks-1, 0, -1):
            self.up_blocks['up_block_%s' % i] = layers.ConvBlock(2 ** i * init_features, 1, int(2 ** (i - 2) * init_features),
                                                                 batch_norm=False, dropout=dropout,
                                                                 activation=activation)

        self.output_conv = nn.Conv3d(init_features//2+in_channels, 3, 3, 1, 1)
        nn.init.normal_(self.output_conv.weight, 0, 0.001)

    def forward(self, x):
        x_enc = [x]
        for i in range(self.num_blocks):
            x_enc.append(self.down_blocks['down_block_%s' % i](x_enc[-1]))

        x = x_enc[-1]
        x = self.bottleneck_block(x)

        for i in range(self.num_blocks-1, 0, -1):
            x = self.upsample(x)
            x = torch.cat([x, x_enc[i]], 1)
            x = self.up_blocks['up_block_%s' % i](x)

        x = self.upsample(x)
        x = torch.cat([x, x_enc[0]], 1)
        y = self.output_conv(x)

        return y


class ResidualEncoder(nn.Module):
    def __init__(self, in_channels, dimension=3, init_features=16, num_blocks=4, norm_type=None,
                 dropout=0, activation=nn.LeakyReLU(), switch_bn=False, **kwargs):
        super(ResidualEncoder, self).__init__()
        self.in_channels = in_channels
        self.dimension = dimension
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

        self.down_blocks = nn.ModuleDict()
        for i in range(self.num_blocks):
            down_in_channels = self.in_channels if i == 0 else 2 ** i * self.init_features
            self.down_blocks['down_block_%s' % i] = layers.ResBlock(down_in_channels,
                                                                    dimension=self.dimension,
                                                                    num_features=2 ** i * self.init_features,
                                                                    norm_type=self.norm_type,
                                                                    dropout=self.dropout,
                                                                    activation=self.activation,
                                                                    switch_bn=self.switch_bn,
                                                                    modalities=self.modalities,
                                                                    track_running_stats=self.tracking_running_stats,
                                                                    momentum=self.momentum)

        self.trans_blocks = nn.ModuleDict()
        for i in range(self.num_blocks):
            self.trans_blocks['trans_block_%s' % i] = layers.ConvBlock(2 ** i * self.init_features,
                                                                       2 ** (i + 1) * self.init_features,
                                                                       dimension=self.dimension,
                                                                       num_layers=1, stride=2,
                                                                       norm_type=self.norm_type,
                                                                       dropout=self.dropout,
                                                                       activation=self.activation,
                                                                       switch_bn=self.switch_bn,
                                                                       modalities=self.modalities,
                                                                       track_running_stats=self.tracking_running_stats,
                                                                       momentum=self.momentum)

        self.bottleneck_block = layers.ResBlock(2 ** self.num_blocks * self.init_features,
                                                dimension=self.dimension,
                                                num_features=2 ** self.num_blocks * self.init_features,
                                                norm_type=self.norm_type, dropout=self.dropout,
                                                activation=self.activation,
                                                switch_bn=self.switch_bn,
                                                modalities=self.modalities,
                                                track_running_stats=self.tracking_running_stats,
                                                momentum=self.momentum)

    def forward(self, x, **kwargs):
        m = kwargs.pop('modality', None)
        if self.switch_bn and x.ndim == self.dimension + 2:
            assert m is not None
        x_enc = []
        for i in range(self.num_blocks):
            x_enc.append(self.down_blocks['down_block_%s' % i](x, modality=m))
            x = self.trans_blocks['trans_block_%s' % i](x_enc[-1], modality=m)

        x = self.bottleneck_block(x, modality=m)

        return x, x_enc


class ResidualDecoder(nn.Module):
    def __init__(self, dimension=3, init_features=16, num_blocks=4, norm_type=None,
                 dropout=0, activation=nn.LeakyReLU(), switch_bn=False, **kwargs):
        super(ResidualDecoder, self).__init__()
        self.dimension = dimension
        self.init_features = init_features
        self.num_blocks = num_blocks
        self.norm_type = norm_type
        self.dropout = dropout
        self.activation = activation
        self.switch_bn = switch_bn
        self.kwargs = kwargs
        self.end_block = self.kwargs.pop('end_block', 0)
        self.modalities = self.kwargs.pop('modalities', None)
        self.tracking_running_stats = self.kwargs.pop('tracking_running_stats', True)
        self.momentum = self.kwargs.pop('momentum', 0.01)
        if self.switch_bn:
            assert self.modalities is not None

        self.upsamples = nn.ModuleDict()
        for i in range(self.end_block, self.num_blocks):
            self.upsamples['upsample_%s' % i] = layers.ResAddUpBlock(2 ** (i + 1) * self.init_features,
                                                                     2 ** i * self.init_features,
                                                                     dimension=self.dimension,
                                                                     norm_type=self.norm_type,
                                                                     dropout=self.dropout,
                                                                     activation=self.activation,
                                                                     track_running_stats=self.tracking_running_stats,
                                                                     momentum=self.momentum)
        self.up_blocks = nn.ModuleDict()
        for i in range(self.end_block, self.num_blocks):
            self.up_blocks['up_block_%s' % i] = layers.ResBlock(2 ** i * self.init_features,
                                                                dimension=self.dimension,
                                                                num_features=2 ** i * self.init_features,
                                                                norm_type=self.norm_type,
                                                                dropout=self.dropout,
                                                                activation=self.activation,
                                                                switch_bn=self.switch_bn,
                                                                modalities=self.modalities,
                                                                track_running_stats=self.tracking_running_stats,
                                                                momentum=self.momentum)

    def forward(self, x, x_enc, **kwargs):
        m = kwargs.pop('modality', None)
        if self.switch_bn and x.ndim == self.dimension + 2:
            assert m is not None
        x_dec = [x]
        for i in range(self.num_blocks - 1, self.end_block - 1, -1):
            x = self.upsamples['upsample_%s' % i](x_dec[-1])
            x += x_enc[i]
            x_dec.append(self.up_blocks['up_block_%s' % i](x, modality=m))

        x_dec = x_dec[::-1]

        return x_dec


class ResidualUNet(nn.Module):
    def __init__(self, in_channels, out_channels=3, dimension=3, init_features=16, num_blocks=4, norm_type=None,
                 dropout=0, activation=nn.LeakyReLU(), **kwargs):
        super(ResidualUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimension = dimension
        self.init_features = init_features
        self.num_blocks = num_blocks
        self.norm_type = norm_type
        self.dropout = dropout
        self.activation = activation
        self.kwargs = kwargs
        self.track_running_stats = self.kwargs.pop('track_running_stats', True)
        self.momentum = self.kwargs.pop('momentum', 0.01)
        self.output_levels = self.kwargs.pop('output_levels', (0, ))
        if self.output_levels == (-1, ):
            self.output_levels = list(range(self.num_blocks))
        self.combine_outputs = self.kwargs.pop('combine_outputs', True)

        self.encoder = ResidualEncoder(self.in_channels,
                                       dimension=self.dimension,
                                       init_features=self.init_features,
                                       num_blocks=self.num_blocks,
                                       norm_type=self.norm_type,
                                       dropout=self.dropout,
                                       activation=self.activation,
                                       track_running_stats=self.track_running_stats,
                                       momentum=self.momentum)

        self.decoder = ResidualDecoder(dimension=self.dimension,
                                       init_features=self.init_features,
                                       num_blocks=self.num_blocks,
                                       norm_type=self.norm_type,
                                       dropout=self.dropout,
                                       actication=self.activation,
                                       track_running_stats=self.track_running_stats,
                                       momentum=self.momentum)

        self.output_convs = nn.ModuleDict()
        for i in self.output_levels:
            if self.dimension == 3:
                self.output_convs['output_conv_%s' % i] = nn.Conv3d(2 ** i * self.init_features, self.out_channels, 1)

            elif self.dimension == 2:
                self.output_convs['output_conv_%s' % i] = nn.Conv2d(2 ** i * self.init_features, self.out_channels, 1)
            else:
                raise NotImplementedError

            nn.init.constant_(self.output_convs['output_conv_%s' % i].weight, 0)
            nn.init.constant_(self.output_convs['output_conv_%s' % i].bias, 0)

    def forward(self, x):
        x, x_enc = self.encoder(x)
        x_dec = self.decoder(x, x_enc)

        if tuple(self.output_levels) == (0, ):
            y = self.output_convs['output_conv_0'](x_dec[0])
        else:
            y = dict([(i, self.output_convs['output_conv_%s' % i](x_dec[i])) for i in self.output_levels])
            if self.combine_outputs:
                out = []
                if self.dimension == 2:
                    mode = 'bilinear'
                elif self.dimension == 3:
                    mode = 'trilinear'
                else:
                    raise NotImplementedError
                for k, v in y.items():
                    out.append(F.interpolate(v, scale_factor=2 ** k, mode=mode, align_corners=False))

                y = torch.sum(torch.stack(out), dim=0)

        return y

