import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
import numpy as np
from utils.cost_volume import costvolume
import time
from .submodule import *
from .aggregation import *

class FastLF_Net(nn.Module):
    def __init__(self, device, maxdisp, k, nums):
        super(FastLF_Net, self).__init__()
        self.device = device
        self.maxdisp = maxdisp // 2 ** k
        self.nums = nums

        self.input_channels = 1
        self.imfeatures = 16

        self.feature_extraction = feature_extraction(self.input_channels, self.imfeatures)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, center_input):
        center_feature, edge_tensor, edge_out = self.feature_extraction(center_input, if_center=True)

        return torch.sigmoid(edge_out)

