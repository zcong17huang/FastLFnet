import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
import numpy as np
from .bam import *

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))
        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)

class SimpleBottleneck(nn.Module):
    """Simple bottleneck block without channel expansion"""
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(SimpleBottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride, dilation)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class feature_extraction(nn.Module):
    def __init__(self, in_c, out_c):
        super(feature_extraction, self).__init__()
        self.inplanes = out_c
        self.firstconv = nn.Sequential(convbn(in_c, out_c, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(out_c, out_c, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True)
                                       )
        self.layer1 = self._make_layer(BasicBlock, out_c*2, 6, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, out_c*4, 2, 1,1,1)
        self.layer3 = self._make_layer(BasicBlock, out_c*4, 2, 2,1,1)
        self.layer4 = self._make_layer(BasicBlock, out_c*4, 2, 2,1,1)
        self.layer5 = self._make_layer(BasicBlock, out_c*4, 2, 2,1,1)

        self.branch3 = nn.Sequential(convbn(out_c*4, out_c, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(convbn(out_c*4, out_c//2, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True))

        self.branch5 = nn.Sequential(convbn(out_c*4, out_c//2, 3, 1, 1, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(out_c*9, out_c*4, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(out_c*4, out_c*2, kernel_size=3, padding=1, bias=False))

        self.bam = BAM(gate_channel=out_c*2, reduction_ratio=2)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        first = self.firstconv(x)

        layer1 = self.layer1(first)     #conv1_x

        layer2 = self.layer2(layer1)    #conv2_x

        layer3 = self.layer3(layer2)    #conv3_x

        layer4 = self.layer4(layer3)    #conv4_x

        layer5 = self.layer5(layer4)    #conv5_x

        output_branch3 = self.branch3(layer3)
        output_branch3 = F.interpolate(output_branch3, (first.size()[2],first.size()[3]),mode='bilinear',align_corners=True)

        output_branch4 = self.branch4(layer4)
        output_branch4 = F.interpolate(output_branch4, (first.size()[2],first.size()[3]),mode='bilinear',align_corners=True)

        output_branch5 = self.branch5(layer5)
        output_branch5 = F.interpolate(output_branch5, (first.size()[2],first.size()[3]),mode='bilinear',align_corners=True)

        x = torch.cat((first, layer1, layer2, output_branch3, output_branch4, output_branch5), 1)
        x = self.lastconv(x)
        x = self.bam(x)

        return x


class disparityregression(nn.Module):
    def __init__(self):
        super(disparityregression, self).__init__()

    def forward(self, x, maxdisp):
        assert x.dim() == 4
        x = F.softmax(x, dim=1)

        disp = torch.arange(-maxdisp, maxdisp + 1).type_as(x)
        disp = disp.view(1, x.size(1), 1, 1)

        pred = torch.sum(x * disp, 1, keepdim=True)
        return pred


class faster_disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(faster_disparityregression, self).__init__()

        self.maxdisp = maxdisp
        self.disp_sample_number = 2 * maxdisp +1

        disp_sample = torch.linspace(-self.maxdisp, self.maxdisp, self.disp_sample_number)
        disp_sample = disp_sample.repeat(1, 1, 1, 1, 1).permute(0, 1, 4, 2, 3).contiguous().cuda()

        self.weight_data = disp_sample

    def forward(self, x):
        assert x.dim() == 4
        x = F.softmax(x, dim=1)

        x = x.unsqueeze(1)

        pred = F.conv3d(x, weight=self.weight_data)
        # [B, 1, 1, W, H] -> [B, 1, W, H]
        pred = pred.squeeze(1)

        return pred

