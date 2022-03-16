import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio, dilation_conv_num=2):
        super(SpatialGate, self).__init__()
        dilation_val = [1, 2]
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3,
                                                                     padding=dilation_val[i], dilation=dilation_val[i]) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        return self.gate_s( in_tensor ).expand_as(in_tensor)

class BAM(nn.Module):
    def __init__(self, gate_channel, reduction_ratio):
        super(BAM, self).__init__()
        self.spatial_att = SpatialGate(gate_channel, reduction_ratio)
    def forward(self,in_tensor):
        att = 1 + torch.sigmoid(self.spatial_att(in_tensor))
        return att * in_tensor
