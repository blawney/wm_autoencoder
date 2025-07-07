import sys

import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def resize_conv1x1(in_planes, out_planes, scale=1):
    """Upsample + 1x1 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv1x1(in_planes, out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv1x1(in_planes, out_planes))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def resize_conv3x3(in_planes, out_planes, scale=1):
    """Upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv3x3(in_planes, out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv3x3(in_planes, out_planes))


def trans_conv1x1(in_planes, out_planes, stride=1, output_padding=0):
    """1x1 convolution."""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, output_padding=output_padding, bias=False)


def trans_conv3x3(in_planes: int, out_planes: int, 
                  stride: int = 1, dilation: int = 1, 
                  output_padding: int = 0):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        output_padding = output_padding,
        bias=False,
        dilation=dilation)


class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate."""

    def __init__(self, size=None, scale_factor=None) -> None:
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)


class DecoderBottleneckV1(nn.Module):
    '''
    Bottleneck module that uses Interpolations followed by
    standard Conv2d operations to increase the size of the 
    layers.
    '''

    expansion = 4

    def __init__(self, inplanes, planes, scale=1, upsample=None):
        super().__init__()
        width = planes

        self.conv1 = resize_conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = resize_conv3x3(width, width, scale)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample
        self.scale = scale

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        return self.relu(out)


class DecoderBottleneckV2(nn.Module):
    '''
    Decoder bottleneck that makes use of ConvTranspose2d
    to increase the size of the layers
    '''

    expansion = 4

    def __init__(self, inplanes, planes, scale=1, upsample=None, output_padding=0):
        super().__init__()
        width = planes

        self.conv1 = trans_conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = trans_conv3x3(width, width, scale, output_padding=output_padding)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = trans_conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample
        self.scale = scale

    def forward(self, x):

        identity = x

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        return self.relu(out)


class ResNet50DecoderV1(nn.Module):
    '''
    An "inverse" ResNet50 that is expected to use DecoderBottleneckV1
    as the block
    '''
    
    def __init__(self, block, latent_dim, input_size):
        super().__init__()

        if type(block) != DecoderBottleneckV1:
            print('Warning- not tested with this combination. We expect the'
                  ' module to be used with blocks of type DecoderBottleneckV1')
            
        layers = [3, 4, 6, 3]
        self.expansion = block.expansion
        self.inplanes = 512 * block.expansion
        self.upscale_factor = 4

        self.linear1 = nn.Linear(latent_dim, self.inplanes * 4 * 4)
        if (input_size % self.upscale_factor) != 0:
            print(f'Given the input size {input_size} and scaling {self.upscale_factor},'
                   ' the reconstructed image will NOT have the same dimensions as the'
                   ' input. Adjust accordingly.')
            sys.exit(1)
        self.upscale1 = Interpolate(size=input_size // self.upscale_factor)

        # the bottleneck layers
        self.layer1 = self._make_layer(block, 256, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 128, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 64, layers[2], scale=1)
        self.layer4 = self._make_layer(block, 64, layers[3])
        self.final_conv = nn.Conv2d(64 * block.expansion, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.initialize_weights()

    def forward(self, x):

        # from 2048 -> 32,768
        x = self.linear1(x)

        # reshape to (B, 2048, 4, 4)
        x = x.view(x.size(0), 512 * self.expansion, 4, 4)

        # if the input_size is 128, then this will do
        # an interpolation to (B, 2048, 3, 3)
        x = self.upscale1(x)

        x = self.layer1(x) # (B, 1024, 6, 6)

        x = self.layer2(x) # (B, 512, 12, 12)

        x = self.layer3(x) # (B, 256, 24, 24)

        x = self.layer4(x) # (B, 256, 24, 24)

        x = self.final_conv(x) # [B, 3, 24, 24])

        return x

    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                
                # Do not initialize bias (due to batchnorm)-
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.BatchNorm2d):
                # Standard initialization for batch normalization-
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


class ResNet50DecoderV2(nn.Module):
    '''
    An "inverse" ResNet50 that makes us of the DecoderBottleneckV2
    bottleneck (which uses transposed convolutions)
    '''
    
    def __init__(self, block, latent_dim, input_size):

        super().__init__()
        layers = [3, 4, 6, 3]
        self.expansion = block.expansion
        self.inplanes = 512 * block.expansion
        self.upscale_factor = 8

        self.linear1 = nn.Linear(latent_dim, self.inplanes * 4 * 4)
        if (input_size % self.upscale_factor) != 0:
            print(f'Given the input size {input_size} and scaling {self.upscale_factor},'
                   ' the reconstructed image will NOT have the same dimensions as the'
                   ' input. Adjust accordingly.')
            sys.exit(1)
        self.upscale1 = Interpolate(size=input_size // self.upscale_factor)

        # the bottleneck layers
        self.layer1 = self._make_layer(block, 256, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 128, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 64, layers[2], scale=2)
        self.layer4 = self._make_layer(block, 32, layers[3], output_padding=0)
        self.final_conv = nn.Conv2d(32 * block.expansion, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.initialize_weights()

    def forward(self, x):

        x = self.linear1(x)

        x = x.view(x.size(0), 512 * self.expansion, 4, 4)

        x = self.upscale1(x)
        x = self.layer1(x)
        x = self.layer2(x) 
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final_conv(x)
        return x

    def _make_layer(self, block, planes, blocks, scale=1, output_padding=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion or output_padding==0:
            upsample = nn.Sequential(
                trans_conv1x1(self.inplanes, planes * block.expansion, scale, output_padding=output_padding),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample, output_padding=output_padding))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                
                # Do not initialize bias (due to batchnorm)-
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.BatchNorm2d):
                # Standard initialization for batch normalization-
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
