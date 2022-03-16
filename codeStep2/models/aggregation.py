from .submodule import *

class OneAggregation(nn.Module):
    def __init__(self, maxdisp, num_blocks):
        super(OneAggregation, self).__init__()

        self.maxdisp = maxdisp
        self.num_blocks = num_blocks
        num_candidates = 2 * maxdisp + 1

        OneBranch = nn.ModuleList()
        for i in range(num_blocks):
            OneBranch.append(SimpleBottleneck(num_candidates, num_candidates))

        self.OneBranch = nn.Sequential(*OneBranch)

    def forward(self, x):
        return self.OneBranch(x)


# Stacked Modules
class MultiAggregation(nn.Module):
    def __init__(self, multi_disp, num_scales, num_blocks, fusion_blocks):
        super(MultiAggregation, self).__init__()

        self.multi_disp = multi_disp
        self.num_scales = num_scales
        self.num_blocks = num_blocks
        self.fusion_blocks = fusion_blocks

        self.branchs1 = OneAggregation(maxdisp = multi_disp[0],
                                        num_blocks = num_blocks)
        self.branchs2 = OneAggregation(maxdisp = multi_disp[1],
                                        num_blocks = num_blocks)
        self.branchs3 = OneAggregation(maxdisp = multi_disp[2],
                                        num_blocks = num_blocks)
        self.branchs4 = OneAggregation(maxdisp = multi_disp[3],
                                        num_blocks = num_blocks)

        self.up1 = nn.Sequential(conv3x3(in_planes = (2*multi_disp[0]+1), out_planes = (2*multi_disp[1]+1)),
                                 nn.BatchNorm2d(2*multi_disp[1]+1))
        self.fuse2 = SimpleBottleneck(inplanes = (2*multi_disp[1]+1), planes = (2*multi_disp[1]+1))

        # ----------------
        self.up2_1 = nn.Sequential(conv3x3(in_planes = (2*multi_disp[1]+1), out_planes = (2*multi_disp[2]+1)),
                                 nn.BatchNorm2d(2*multi_disp[2]+1))
        self.up2_2 = nn.Sequential(conv3x3(in_planes = (2*multi_disp[1]+1), out_planes = (2*multi_disp[2]+1)),
                                 nn.BatchNorm2d(2*multi_disp[2]+1))

        self.fuse3_1 = SimpleBottleneck(inplanes = (2*multi_disp[2]+1), planes = (2*multi_disp[2]+1))
        self.fuse3_2 = SimpleBottleneck(inplanes = (2*multi_disp[2]+1), planes = (2*multi_disp[2]+1))

        # ----------------
        self.up3_1 = nn.Sequential(conv3x3(in_planes = (2*multi_disp[2]+1), out_planes = (2*multi_disp[3]+1)),
                                 nn.BatchNorm2d(2*multi_disp[3]+1))
        self.up3_2 = nn.Sequential(conv3x3(in_planes = (2*multi_disp[2]+1), out_planes = (2*multi_disp[3]+1)),
                                 nn.BatchNorm2d(2*multi_disp[3]+1))
        self.up3_3 = nn.Sequential(conv3x3(in_planes = (2*multi_disp[2]+1), out_planes = (2*multi_disp[3]+1)),
                                 nn.BatchNorm2d(2*multi_disp[3]+1))

        self.fuse4_1 = SimpleBottleneck(inplanes = (2*multi_disp[3]+1), planes = (2*multi_disp[3]+1))
        self.fuse4_2 = SimpleBottleneck(inplanes = (2*multi_disp[3]+1), planes = (2*multi_disp[3]+1))
        self.fuse4_3 = SimpleBottleneck(inplanes = (2*multi_disp[3]+1), planes = (2*multi_disp[3]+1))

        # ----------------
        num_candidates = 2 * multi_disp[-1] + 1
        fusions = nn.ModuleList()
        for i in range(fusion_blocks):
            fusions.append(SimpleBottleneck(num_candidates, num_candidates))
        self.fusions = nn.Sequential(*fusions)


        in_channels = 2 * multi_disp[-1]+1
        out_channels = in_channels
        self.classify = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, cost1, cost2, cost3, cost4):
        assert self.num_scales == 4

        cost1 = self.branchs1(cost1)
        cost2 = self.branchs2(cost2)
        cost3 = self.branchs3(cost3)
        cost4 = self.branchs4(cost4)

        # ----------------
        cost2_1 = cost2 + self.up1(cost1)
        cost2_1 = self.fuse2(cost2_1)

        # ----------------
        cost3_1 = cost3 + self.up2_1(cost2)
        cost3_1 = self.fuse3_1(cost3_1)

        cost3_2 = cost3_1 + self.up2_2(cost2_1)
        cost3_2 = self.fuse3_2(cost3_2)

        # ----------------
        cost4 = cost4 + self.up3_1(cost3)
        cost4 = self.fuse4_1(cost4)

        cost4 = cost4 + self.up3_2(cost3_1)
        cost4 = self.fuse4_2(cost4)

        cost4 = cost4 + self.up3_3(cost3_2)
        cost4 = self.fuse4_3(cost4)

        # one branch
        out = self.fusions(cost4)

        out = self.classify(out)

        return out



