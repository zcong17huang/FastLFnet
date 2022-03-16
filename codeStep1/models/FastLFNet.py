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

        self.multi_disp = []
        for i in range(4):
            self.multi_disp.append((i + 1) * (self.maxdisp // 4))           # [4,8,12,16]

        self.feature_extraction = feature_extraction(self.input_channels, self.imfeatures)

        self.cost_volume_construct = costvolume(self.device, self.maxdisp)

        # Cost aggregation
        self.aggregation = MultiAggregation(multi_disp=self.multi_disp,
                                            num_scales=4,
                                            num_blocks=1,
                                            fusion_blocks=4)

        self.disparityregression = disparityregression()
        # self.disparityregression = faster_disparityregression(maxdisp=self.multi_disp[-1])

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


    def forward(self, center_input, view1_input, view2_input, view3_input, view4_input):
        H = center_input.shape[2]
        W = center_input.shape[3]

        center_feature = self.feature_extraction(center_input)
        view1_list = self.loop_extract_feature(view1_input, num_view=4)
        view2_list = self.loop_extract_feature(view2_input, num_view=4)
        view3_list = self.loop_extract_feature(view3_input, num_view=4)
        view4_list = self.loop_extract_feature(view4_input, num_view=4)

        cost_out1, cost_out2,\
        cost_out3, cost_out4 = self.cost_volume_construct(center_feature, view1_list, view2_list, view3_list, view4_list)

        cost = self.aggregation(cost_out1, cost_out2, cost_out3, cost_out4)

        pred  = self.disparityregression(cost, self.multi_disp[-1])
        # pred = self.disparityregression(cost)

        pred  = pred  / 4.  # revert to disparity range of [-4, 4]

        b,d,h,w = cost.shape
        assert d == (2 * self.maxdisp + 1) and h == H and w == W, 'NO downsample!!! Dimension of cost error!!!'

        return pred


    def loop_extract_feature(self, input_batch, num_view):
        feature_list = []
        for i in range(num_view):
            one_im = torch.unsqueeze(input_batch[:, i, :, :], 1)
            feature_list.append(self.feature_extraction(one_im))
        return feature_list
