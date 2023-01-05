# -*- coding: utf-8 -*-
"""
Network blocks and layers for network construction.

@author: Xinzhe Luo
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResBlock(nn.Module):
    def __init__(self, in_channels, num_features=16, dimension=3, num_layers=2, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, norm_type='batch', downsample=False, dropout=0.,
                 activation=nn.LeakyReLU(), switch_bn=False, **norm_kwargs):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.dimension = dimension
        self.num_layers = num_layers
        self.num_features = self._check_input_as_tuple(num_features)
        assert len(self.num_features) == self.num_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.norm_type = norm_type
        self.downsample = downsample
        self.dropout_rate = dropout
        self.activation = activation
        self.switch_bn = switch_bn
        self.norm_kwargs = norm_kwargs
        self.modalities = self.norm_kwargs.pop('modalities', None)
        if self.switch_bn:
            assert self.modalities is not None

        if self.dimension == 3:
            self.Conv = nn.Conv3d
            self.Dropout = nn.Dropout3d
            if self.downsample:
                self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        elif self.dimension == 2:
            self.Conv = nn.Conv2d
            self.Dropout = nn.Dropout2d
            if self.downsample:
                self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError

        self.convs = nn.ModuleDict()
        self.dropouts = nn.ModuleDict()
        self.convs['conv_0'] = self.Conv(self.in_channels, self.num_features[0], self.kernel_size, self.stride,
                                         self.padding, self.dilation, self.groups, self.bias)
        self.dropouts['dropout_0'] = self.Dropout(p=self.dropout_rate)
        for i in range(1, self.num_layers):
            self.convs['conv_%s' % i] = self.Conv(self.num_features[i-1], self.num_features[i], self.kernel_size,
                                                  self.stride, self.padding, self.dilation, self.groups, self.bias)
            self.dropouts['dropout_%s' % i] = self.Dropout(p=self.dropout_rate)

        if self.in_channels != self.num_features[-1]:
            self.res_conv = self.Conv(self.in_channels, self.num_features[-1], 1, 1, dilation=self.dilation,
                                      groups=self.groups, bias=self.bias)

        if self.norm_type:
            self.res_norm = nn.ModuleDict(zip(self.modalities,
                                              [Normalize(self.num_features[-1],
                                                         dimension=self.dimension,
                                                         norm_type=self.norm_type,
                                                         **self.norm_kwargs) for _ in range(len(self.modalities))])) \
                if self.switch_bn else Normalize(self.num_features[-1], dimension=self.dimension,
                                                 norm_type=self.norm_type, **self.norm_kwargs)
            self.norms = nn.ModuleDict()
            for i in range(self.num_layers):
                self.norms['%s_norm_%s' % (self.norm_type, i)] = \
                    nn.ModuleDict(zip(self.modalities,
                                      [Normalize(self.num_features[i],
                                       dimension=self.dimension,
                                       norm_type=self.norm_type,
                                       **self.norm_kwargs) for _ in range(len(self.modalities))])) \
                        if self.switch_bn else Normalize(self.num_features[i],
                                                         dimension=self.dimension,
                                                         norm_type=self.norm_type,
                                                         **self.norm_kwargs)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight,
                                        a=(self.activation.negative_slope
                                           if isinstance(self.activation, nn.LeakyReLU) else 0))

    def forward(self, x, **kwargs):
        mm = (x.ndim == self.dimension + 3)
        if self.switch_bn and x.ndim == self.dimension + 2:
            m = kwargs['modality']

        y = x.clone()
        for i in range(self.num_layers):
            if mm:
                y = rearrange(y, 'B M ... -> (B M) ...')
            y = self.convs['conv_%s' % i](y)
            if mm:
                y = torch.unbind(rearrange(y, '(B M) ... -> B M ...', M=len(self.modalities)),
                                 dim=1)
            if self.norm_type:
                if mm:
                    if self.switch_bn:
                        y = [self.norms['%s_norm_%s' % (self.norm_type, i)][self.modalities[k]](y[k])
                             for k in range(len(self.modalities))]
                    else:
                        y = [self.norms['%s_norm_%s' % (self.norm_type, i)](y[k])
                             for k in range(len(self.modalities))]
                else:
                    y = self.norms['%s_norm_%s' % (self.norm_type, i)][m](y) if self.switch_bn \
                        else self.norms['%s_norm_%s' % (self.norm_type, i)](y)

            if i < self.num_layers - 1:
                if self.activation:
                    if mm:
                        y = [self.activation(y[k]) for k in range(len(self.modalities))]
                    else:
                        y = self.activation(y)

                if mm:
                    y = [self.dropouts['dropout_%s' % i](y[k]) for k in range(len(self.modalities))]
                else:
                    y = self.dropouts['dropout_%s' % i](y)
            if mm:
                y = torch.stack(y, dim=1)
        if mm:
            y = torch.unbind(y, dim=1)

        if self.in_channels != self.num_features[-1]:
            if mm:
                x = rearrange(x, 'B M ... -> (B M) ...')
            x = self.res_conv(x)
            if mm:
                x = rearrange(x, '(B M) ... -> B M ...', M=len(self.modalities))

        if mm:
            x = torch.unbind(x, dim=1)

        if self.norm_type:
            if mm:
                if self.switch_bn:
                    x = [self.res_norm[self.modalities[k]](x[k]) for k in range(len(self.modalities))]
                else:
                    x = [self.res_norm(x[k]) for k in range(len(self.modalities))]
            else:
                x = self.res_norm[m](x) if self.switch_bn else self.res_norm(x)

        if mm:
            y = [y[k] + x[k] for k in range(len(self.modalities))]
        else:
            y += x
        if self.activation:
            if mm:
                y = [self.activation(y[k]) for k in range(len(self.modalities))]
            else:
                y = self.activation(y)

        if self.downsample:
            if mm:
                y = [self.maxpool(y[k]) for k in range(len(self.modalities))]
            else:
                y = self.maxpool(y)

        if mm:
            y = torch.stack(y, dim=1)
        return y

    def _check_input_as_tuple(self, input):
        if isinstance(input, (list, tuple)):
            return tuple(input)
        elif isinstance(input, int):
            return (input, ) * self.num_layers
        else:
            raise NotImplementedError


class ConvBlock(nn.Module):
    def __init__(self, in_channels, num_features=16, dimension=3, num_layers=2, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False, norm_type='batch', downsample=False, dropout=0.,
                 activation=nn.LeakyReLU(), switch_bn=False, **norm_kwargs):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.dimension = dimension
        self.num_layers = num_layers
        self.num_features = self._check_input_as_tuple(num_features)
        assert len(self.num_features) == self.num_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.norm_type = norm_type
        self.downsample = downsample
        self.dropout_rate = dropout
        self.activation = activation
        self.switch_bn = switch_bn
        self.norm_kwargs = norm_kwargs
        self.modalities = self.norm_kwargs.pop('modalities', None)
        if self.switch_bn:
            assert self.modalities is not None

        if self.dimension == 3:
            self.Conv = nn.Conv3d
            self.Dropout = nn.Dropout3d
            if self.downsample:
                self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        elif self.dimension == 2:
            self.Conv = nn.Conv2d
            self.Dropout = nn.Dropout2d
            if self.downsample:
                self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError

        self.convs = nn.ModuleDict()
        self.dropouts = nn.ModuleDict()
        self.convs['conv_0'] = self.Conv(self.in_channels, self.num_features[0], self.kernel_size, self.stride,
                                         self.padding, self.dilation, self.groups, self.bias)
        self.dropouts['dropout_0'] = self.Dropout(p=self.dropout_rate)
        for i in range(1, self.num_layers):
            self.convs['conv_%s' % i] = self.Conv(self.num_features[i-1], self.num_features[i], self.kernel_size,
                                                  self.stride, self.padding, self.dilation, self.groups, self.bias)
            self.dropouts['dropout_%s' % i] = self.Dropout(p=self.dropout_rate)

        if self.norm_type:
            self.norms = nn.ModuleDict()
            for i in range(self.num_layers):
                self.norms['%s_norm_%s' % (self.norm_type, i)] = \
                    nn.ModuleDict(zip(self.modalities,
                                      [Normalize(self.num_features[i],
                                                 dimension=self.dimension,
                                                 norm_type=self.norm_type,
                                                 **self.norm_kwargs) for _ in range(len(self.modalities))])) \
                        if self.switch_bn else Normalize(self.num_features[i],
                                                         dimension=self.dimension,
                                                         norm_type=self.norm_type,
                                                         **self.norm_kwargs)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight,
                                        a=(self.activation.negative_slope
                                           if isinstance(self.activation, nn.LeakyReLU) else 0))

    def forward(self, x, **kwargs):
        mm = (x.ndim == self.dimension + 3)
        if self.switch_bn and x.ndim == self.dimension + 2:
            m = kwargs['modality']

        y = x.clone()
        for i in range(self.num_layers):
            if mm:
                y = rearrange(y, 'B M ... -> (B M) ...')
            y = self.convs['conv_%s' % i](y)
            if mm:
                y = torch.unbind(rearrange(y, '(B M) ... -> B M ...', M=len(self.modalities)),
                                 dim=1)
            if self.norm_type:
                if mm:
                    if self.switch_bn:
                        y = [self.norms['%s_norm_%s' % (self.norm_type, i)][self.modalities[k]](y[k])
                             for k in range(len(self.modalities))]
                    else:
                        y = [self.norms['%s_norm_%s' % (self.norm_type, i)](y[k])
                             for k in range(len(self.modalities))]
                else:
                    y = self.norms['%s_norm_%s' % (self.norm_type, i)][m](y) if self.switch_bn \
                        else self.norms['%s_norm_%s' % (self.norm_type, i)](y)

            if self.activation:
                if mm:
                    y = [self.activation(y[k]) for k in range(len(self.modalities))]
                else:
                    y = self.activation(y)

            if mm:
                y = [self.dropouts['dropout_%s' % i](y[k]) for k in range(len(self.modalities))]
            else:
                y = self.dropouts['dropout_%s' % i](y)

            if mm:
                y = torch.stack(y, dim=1)

        if mm:
            y = torch.unbind(y, dim=1)

        if self.downsample:
            if mm:
                y = [self.maxpool(y[k]) for k in range(len(self.modalities))]
            else:
                y = self.maxpool(y)

        if mm:
            y = torch.stack(y, dim=1)
        return y

    def _check_input_as_tuple(self, input):
        if isinstance(input, (list, tuple)):
            return tuple(input)
        elif isinstance(input, int):
            return (input,) * self.num_layers
        else:
            raise NotImplementedError


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dimension=3, kernel_size=2, stride=2, padding=0, groups=1, bias=False,
                 norm_type='batch', dropout=0., activation=nn.LeakyReLU(), **norm_kwargs):
        super(DeconvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimension = dimension
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.norm_type = norm_type
        self.dropout_rate = dropout
        self.activation = activation
        self.norm_kwargs = norm_kwargs

        if self.dimension == 3:
            self.ConvTranspose = nn.ConvTranspose3d
            self.Dropout = nn.Dropout3d
        elif self.dimension == 2:
            self.ConvTranspose = nn.ConvTranspose2d
            self.Dropout = nn.Dropout2d
        else:
            raise NotImplementedError

        self.deconv = self.ConvTranspose(self.in_channels, self.out_channels, self.kernel_size, self.stride,
                                         self.padding, groups=self.groups, bias=self.bias)
        self.dropout = self.Dropout(p=self.dropout_rate)
        nn.init.kaiming_normal_(self.deconv.weight,
                                a=(self.activation.negative_slope if isinstance(self.activation, nn.LeakyReLU) else 0))
        if self.norm_type:
            self.norm = Normalize(self.out_channels, dimension=self.dimension, norm_type=self.norm_type,
                                  **self.norm_kwargs)

    def forward(self, x):
        y = self.deconv(x)
        if self.norm_type:
            y = self.norm(y)
        if self.activation:
            y = self.activation(y)
        y = self.dropout(y)
        return y


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dimension=3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 bias=False, pool_size=2, norm_type='batch', dropout=0.1, activation=nn.LeakyReLU(), **norm_kwargs):
        super(TransitionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimension = dimension
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.pool_size = pool_size
        self.norm_type = norm_type
        self.dropout_rate = dropout
        self.activation = activation
        self.norm_kwargs = norm_kwargs

        if self.dimension == 3:
            self.Conv = nn.Conv3d
            self.Dropout = nn.Dropout3d
            self.maxpool = nn.MaxPool3d(kernel_size=self.pool_size, stride=self.pool_size)
        elif self.dimension == 2:
            self.Conv = nn.Conv2d
            self.Dropout = nn.Dropout2d
            self.maxpool = nn.MaxPool2d(kernel_size=self.pool_size, stride=self.pool_size)
        else:
            raise NotImplementedError

        self.conv = self.Conv(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding,
                              self.dilation, self.groups, self.bias)
        nn.init.kaiming_normal_(self.conv.weight,
                                a=(self.activation.negative_slope if isinstance(self.activation, nn.LeakyReLU) else 0))
        self.dropout = self.Dropout(p=self.dropout_rate)

        if self.norm_type:
            self.norm = Normalize(self.out_channels, norm_type=self.norm_type, **self.norm_kwargs)

    def forward(self, x):
        y = self.maxpool(x)
        y = self.conv(y)

        if self.norm_type:
            y = self.norm(y)

        y = self.dropout(self.activation(y))

        return y


class LinearAdditiveUpsample(nn.Module):
    def __init__(self, scale_factor=2, chunks=2, dimension=3):
        super(LinearAdditiveUpsample, self).__init__()
        self.dimension = dimension
        self.scale_factor = scale_factor
        self.chunks = chunks

        if self.dimension == 3:
            self.upsample = nn.Upsample(scale_factor=self.scale_factor, align_corners=False, mode='trilinear')
        elif self.dimension == 2:
            self.upsample = nn.Upsample(scale_factor=self.scale_factor, align_corners=False, mode='bilinear')
        else:
            raise NotImplementedError

    def forward(self, x):
        assert x.size(1) % self.chunks == 0, "Number of input features should be divisible by chunks!"
        y = self.upsample(x)
        y = torch.chunk(y, self.chunks, dim=1)
        y = torch.stack(y).sum(dim=0)

        return y


class ResAddUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dimension=3, kernel_size=2, stride=2, padding=0, groups=1, bias=False,
                 norm_type='batch', dropout=0., activation=nn.LeakyReLU(), chunks=2, **norm_kwargs):
        super(ResAddUpBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimension = dimension
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.norm_type = norm_type
        self.dropout = dropout
        self.activation = activation
        self.chunks = chunks
        self.norm_kwargs = norm_kwargs

        self.deconv = DeconvBlock(self.in_channels, self.out_channels, self.dimension, self.kernel_size, self.stride,
                                  self.padding, self.groups, self.bias, self.norm_type, self.dropout, None)
        self.add_module('deconv', self.deconv)
        self.upsample = LinearAdditiveUpsample(self.stride, self.chunks, self.dimension)
        self.add_module('linear_additive_upsample', self.upsample)
        if self.norm_type:
            self.norm = Normalize(self.out_channels, dimension=self.dimension,
                                  norm_type=self.norm_type, **self.norm_kwargs)

    def forward(self, x):
        reshape = False
        if x.ndim == self.dimension + 3:
            reshape = True
            M = x.shape[1]
            x = rearrange(x, 'B M ... -> (B M) ...')
        y = self.deconv(x)
        if self.norm_type:
            y = self.norm(y)
        y += self.upsample(x)
        y = self.activation(y)

        if reshape:
            y = rearrange(y, '(B M) ... -> B M ...', M=M)

        return y


class ConvUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, dimension=3, kernel_size=3, stride=1, padding=1,
                 **conv_kwargs):
        super(ConvUpBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.dimension = dimension
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_kwargs = conv_kwargs

        if self.dimension == 3:
            self.Conv = nn.Conv3d
        elif self.dimension == 2:
            self.Conv = nn.Conv2d
        else:
            raise NotImplementedError

        self.conv = self.Conv(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding,
                              **self.conv_kwargs)
        nn.init.normal_(self.conv.weight, std=0.0001)
        if self.dimension == 3:
            self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='trilinear')
        elif self.dimension == 2:
            self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')

    def forward(self, x):
        y = self.conv(x)
        y = self.upsample(y)

        return y


class AbstractFusion(nn.Module):
    def forward(self, x):
        std, mean = torch.std_mean(x, dim=1, unbiased=True)

        return torch.cat([mean, std.square()], dim=1)


class Normalize(nn.Module):
    def __init__(self, num_features, dimension=3, norm_type='batch', affine=True, **kwargs):
        super(Normalize, self).__init__()
        self.norm_type = norm_type
        self.num_features = num_features
        self.dimension = dimension
        self.affine = affine
        self.kwargs = kwargs
        if self.norm_type is None:
            pass
        elif self.norm_type == 'batch':
            if self.dimension == 3:
                self.norm = nn.BatchNorm3d(self.num_features, affine=self.affine, **self.kwargs)
            elif self.dimension == 2:
                self.norm = nn.BatchNorm2d(self.num_features, affine=self.affine, **self.kwargs)
            else:
                raise NotImplementedError
        elif self.norm_type == 'instance':
            if self.dimension == 3:
                self.norm = nn.InstanceNorm3d(self.num_features, affine=self.affine, **self.kwargs)
            elif self.dimension == 2:
                self.norm = nn.InstanceNorm2d(self.num_features, affine=self.affine, **self.kwargs)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if self.affine:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, *args, **kwargs):
        if self.norm_type is None:
            return x
        elif self.norm_type == 'regional':
            return self.norm(x, *args, **kwargs)
        else:
            return self.norm(x)

