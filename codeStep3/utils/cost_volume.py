import torch
import numpy as np
import torch.nn as nn
import torchvision
import os
from models.submodule import *

class costvolume(nn.Module):
    def __init__(self, device, maxdisp):
        super(costvolume, self).__init__()
        self.device = device
        self.maxdisp = maxdisp


    def forward(self, center_feature, view1_list, view2_list, view3_list, view4_list):
        b, c, h, w = center_feature.size()

        disp1 = 1 * (self.maxdisp // 4)
        cost_out1 = center_feature.new_zeros(b, (2 * disp1 + 1), h, w)
        # view1
        for i in range(4):
            cost = center_feature.new_zeros(b, (2 * disp1 + 1), h, w)

            for m in range(-disp1, disp1 + 1):
                dd = abs(m)
                # feature_0d
                if i % 2 == 0:
                    surround_feature = view1_list[i]
                    if m == 0:
                        cost[:, m + disp1, :, :] = (center_feature*surround_feature).mean(dim=1)

                    elif m > 0:
                        if dd == 0:
                            cost[:, m + disp1, :, :] = (center_feature*surround_feature).mean(dim=1)
                        else:
                            if i < 2:
                                cost[:, m + disp1, :, :-dd] = (center_feature[:, :, :, :-dd]*surround_feature[:, :, :, dd:]).mean(dim=1)
                            else:
                                cost[:, m + disp1, :, dd:] = (center_feature[:, :, :, dd:]*surround_feature[:, :, :, :-dd]).mean(dim=1)
                    else:
                        if dd == 0:
                            cost[:, m + disp1, :, :] = (center_feature*surround_feature).mean(dim=1)
                        else:
                            if i < 2:
                                cost[:, m + disp1, :, dd:] = (center_feature[:, :, :, dd:]*surround_feature[:, :, :, :-dd]).mean(dim=1)
                            else:
                                cost[:, m + disp1, :, :-dd] = (center_feature[:, :, :, :-dd]*surround_feature[:, :, :, dd:]).mean(dim=1)
                # feature_90d
                elif i % 2 == 1:
                    surround_feature = view1_list[i]
                    if m == 0:
                        cost[:, m + disp1, :, :] = (center_feature*surround_feature).mean(dim=1)

                    elif m > 0:
                        if dd == 0:
                            cost[:, m + disp1, :, :] = (center_feature*surround_feature).mean(dim=1)
                        else:
                            if i < 2:
                                cost[:, m + disp1, dd:, :] = (center_feature[:, :, dd:, :]*surround_feature[:, :, :-dd, :]).mean(dim=1)
                            else:
                                cost[:, m + disp1, :-dd, :] = (center_feature[:, :, :-dd, :]*surround_feature[:, :, dd:, :]).mean(dim=1)
                    else:
                        if dd == 0:
                            cost[:, m + disp1, :, :] = (center_feature*surround_feature).mean(dim=1)
                        else:
                            if i < 2:
                                cost[:, m + disp1, :-dd, :] = (center_feature[:, :, :-dd, :]*surround_feature[:, :, dd:, :]).mean(dim=1)
                            else:
                                cost[:, m + disp1, dd:, :] = (center_feature[:, :, dd:, :]*surround_feature[:, :, :-dd, :]).mean(dim=1)
                else:
                    print('cost error !!!!!!')

            cost = cost.contiguous()
            cost_out1 = cost_out1 + cost

        cost_out1 = cost_out1 / 4
        cost_out1 = cost_out1.contiguous()


        disp2 = 2 * (self.maxdisp // 4)
        cost_out2 = center_feature.new_zeros(b, (2 * disp2 + 1), h, w)
        # view2
        for i in range(4):
            cost = center_feature.new_zeros(b, (2 * disp2 + 1), h, w)

            for m in range(-disp2, disp2 + 1):
                dd = abs(m)
                # feature_0d
                if i % 2 == 0:
                    surround_feature = view2_list[i]
                    if m == 0:
                        cost[:, m + disp2, :, :] = (center_feature*surround_feature).mean(dim=1)

                    elif m > 0:
                        if dd == 0:
                            cost[:, m + disp2, :, :] = (center_feature*surround_feature).mean(dim=1)
                        else:
                            if i < 2:
                                cost[:, m + disp2, :, :-dd] = (center_feature[:, :, :, :-dd]*surround_feature[:, :, :, dd:]).mean(dim=1)
                            else:
                                cost[:, m + disp2, :, dd:] = (center_feature[:, :, :, dd:]*surround_feature[:, :, :, :-dd]).mean(dim=1)
                    else:
                        if dd == 0:
                            cost[:, m + disp2, :, :] = (center_feature*surround_feature).mean(dim=1)
                        else:
                            if i < 2:
                                cost[:, m + disp2, :, dd:] = (center_feature[:, :, :, dd:]*surround_feature[:, :, :, :-dd]).mean(dim=1)
                            else:
                                cost[:, m + disp2, :, :-dd] = (center_feature[:, :, :, :-dd]*surround_feature[:, :, :, dd:]).mean(dim=1)
                # feature_90d
                elif i % 2 == 1:
                    surround_feature = view2_list[i]
                    if m == 0:
                        cost[:, m + disp2, :, :] = (center_feature*surround_feature).mean(dim=1)

                    elif m > 0:
                        if dd == 0:
                            cost[:, m + disp2, :, :] = (center_feature*surround_feature).mean(dim=1)
                        else:
                            if i < 2:
                                cost[:, m + disp2, dd:, :] = (center_feature[:, :, dd:, :]*surround_feature[:, :, :-dd, :]).mean(dim=1)
                            else:
                                cost[:, m + disp2, :-dd, :] = (center_feature[:, :, :-dd, :]*surround_feature[:, :, dd:, :]).mean(dim=1)
                    else:
                        if dd == 0:
                            cost[:, m + disp2, :, :] = (center_feature*surround_feature).mean(dim=1)
                        else:
                            if i < 2:
                                cost[:, m + disp2, :-dd, :] = (center_feature[:, :, :-dd, :]*surround_feature[:, :, dd:, :]).mean(dim=1)
                            else:
                                cost[:, m + disp2, dd:, :] = (center_feature[:, :, dd:, :]*surround_feature[:, :, :-dd, :]).mean(dim=1)
                else:
                    print('cost error !!!!!!')

            cost = cost.contiguous()
            cost_out2 = cost_out2 + cost

        cost_out2 = cost_out2 / 4
        cost_out2 = cost_out2.contiguous()


        disp3 = 3 * (self.maxdisp // 4)
        cost_out3 = center_feature.new_zeros(b, (2 * disp3 + 1), h, w)
        # view3
        for i in range(4):
            cost = center_feature.new_zeros(b, (2 * disp3 + 1), h, w)

            for m in range(-disp3, disp3 + 1):
                dd = abs(m)
                # feature_0d
                if i % 2 == 0:
                    surround_feature = view3_list[i]
                    if m == 0:
                        cost[:, m + disp3, :, :] = (center_feature*surround_feature).mean(dim=1)

                    elif m > 0:
                        if dd == 0:
                            cost[:, m + disp3, :, :] = (center_feature*surround_feature).mean(dim=1)
                        else:
                            if i < 2:
                                cost[:, m + disp3, :, :-dd] = (center_feature[:, :, :, :-dd]*surround_feature[:, :, :, dd:]).mean(dim=1)
                            else:
                                cost[:, m + disp3, :, dd:] = (center_feature[:, :, :, dd:]*surround_feature[:, :, :, :-dd]).mean(dim=1)
                    else:
                        if dd == 0:
                            cost[:, m + disp3, :, :] = (center_feature*surround_feature).mean(dim=1)
                        else:
                            if i < 2:
                                cost[:, m + disp3, :, dd:] = (center_feature[:, :, :, dd:]*surround_feature[:, :, :, :-dd]).mean(dim=1)
                            else:
                                cost[:, m + disp3, :, :-dd] = (center_feature[:, :, :, :-dd]*surround_feature[:, :, :, dd:]).mean(dim=1)
                # feature_90d
                elif i % 2 == 1:
                    surround_feature = view3_list[i]
                    if m == 0:
                        cost[:, m + disp3, :, :] = (center_feature*surround_feature).mean(dim=1)

                    elif m > 0:
                        if dd == 0:
                            cost[:, m + disp3, :, :] = (center_feature*surround_feature).mean(dim=1)
                        else:
                            if i < 2:
                                cost[:, m + disp3, dd:, :] = (center_feature[:, :, dd:, :]*surround_feature[:, :, :-dd, :]).mean(dim=1)
                            else:
                                cost[:, m + disp3, :-dd, :] = (center_feature[:, :, :-dd, :]*surround_feature[:, :, dd:, :]).mean(dim=1)
                    else:
                        if dd == 0:
                            cost[:, m + disp3, :, :] = (center_feature*surround_feature).mean(dim=1)
                        else:
                            if i < 2:
                                cost[:, m + disp3, :-dd, :] = (center_feature[:, :, :-dd, :]*surround_feature[:, :, dd:, :]).mean(dim=1)
                            else:
                                cost[:, m + disp3, dd:, :] = (center_feature[:, :, dd:, :]*surround_feature[:, :, :-dd, :]).mean(dim=1)
                else:
                    print('cost error !!!!!!')

            cost = cost.contiguous()
            cost_out3 = cost_out3 + cost

        cost_out3 = cost_out3 / 4
        cost_out3 = cost_out3.contiguous()


        disp4 = 4 * (self.maxdisp // 4)
        cost_out4 = center_feature.new_zeros(b, (2 * disp4 + 1), h, w)
        # view4
        for i in range(4):
            cost = center_feature.new_zeros(b, (2 * disp4 + 1), h, w)

            for m in range(-disp4, disp4 + 1):
                dd = abs(m)
                # feature_0d
                if i % 2 == 0:
                    surround_feature = view4_list[i]
                    if m == 0:
                        cost[:, m + disp4, :, :] = (center_feature*surround_feature).mean(dim=1)

                    elif m > 0:
                        if dd == 0:
                            cost[:, m + disp4, :, :] = (center_feature*surround_feature).mean(dim=1)
                        else:
                            if i < 2:
                                cost[:, m + disp4, :, :-dd] = (center_feature[:, :, :, :-dd]*surround_feature[:, :, :, dd:]).mean(dim=1)
                            else:
                                cost[:, m + disp4, :, dd:] = (center_feature[:, :, :, dd:]*surround_feature[:, :, :, :-dd]).mean(dim=1)
                    else:
                        if dd == 0:
                            cost[:, m + disp4, :, :] = (center_feature*surround_feature).mean(dim=1)
                        else:
                            if i < 2:
                                cost[:, m + disp4, :, dd:] = (center_feature[:, :, :, dd:]*surround_feature[:, :, :, :-dd]).mean(dim=1)
                            else:
                                cost[:, m + disp4, :, :-dd] = (center_feature[:, :, :, :-dd]*surround_feature[:, :, :, dd:]).mean(dim=1)
                # feature_90d
                elif i % 2 == 1:
                    surround_feature = view4_list[i]
                    if m == 0:
                        cost[:, m + disp4, :, :] = (center_feature*surround_feature).mean(dim=1)

                    elif m > 0:
                        if dd == 0:
                            cost[:, m + disp4, :, :] = (center_feature*surround_feature).mean(dim=1)
                        else:
                            if i < 2:
                                cost[:, m + disp4, dd:, :] = (center_feature[:, :, dd:, :]*surround_feature[:, :, :-dd, :]).mean(dim=1)
                            else:
                                cost[:, m + disp4, :-dd, :] = (center_feature[:, :, :-dd, :]*surround_feature[:, :, dd:, :]).mean(dim=1)
                    else:
                        if dd == 0:
                            cost[:, m + disp4, :, :] = (center_feature*surround_feature).mean(dim=1)
                        else:
                            if i < 2:
                                cost[:, m + disp4, :-dd, :] = (center_feature[:, :, :-dd, :]*surround_feature[:, :, dd:, :]).mean(dim=1)
                            else:
                                cost[:, m + disp4, dd:, :] = (center_feature[:, :, dd:, :]*surround_feature[:, :, :-dd, :]).mean(dim=1)
                else:
                    print('cost error !!!!!!')

            cost = cost.contiguous()
            cost_out4 = cost_out4 + cost

        cost_out4 = cost_out4 / 4
        cost_out4 = cost_out4.contiguous()

        return cost_out1, cost_out2, cost_out3, cost_out4



