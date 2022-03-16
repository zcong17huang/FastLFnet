# -*- coding: UTF-8 -*-
import argparse
import os
import numpy as np
import time
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision
import torch.nn.functional as F
import utils.readpfm as rp
from dataloader import listLFfile as lf
from dataloader import HCILoader as DA
from models.FastLFNet import FastLF_Net as FastLFnet

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='FastLFNet')
parser.add_argument("--batchsize", type=int, default=1)
parser.add_argument('--datapath', default='/data1/Dataset/HCI_Benchmark/', help='datapath')
parser.add_argument('--loadmodel', default='./checkpoints/FastLFnet_model_final.tar', help='load model')
parser.add_argument('--outpath', default='./outputs/FastLFnet_out', help='output path')
parser.add_argument('--nums', default=9, type=int, help='nums of NxN')
parser.add_argument('--maxdisp', type=int, default=16, help='maxium disparity')
parser.add_argument('--k', type=int, default=0, help='downsample layers')

parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()

def evaluate(args, device):
    im_name_all = lf.submission_dataloader(args.datapath)
    im_name_all.sort()

    TestImgLoader = torch.utils.data.DataLoader(DA.submissonFloder(im_name_all, args.nums),
                                    batch_size=args.batchsize, num_workers=8, pin_memory=True, drop_last=True)

    model = FastLFnet(device=device, maxdisp=args.maxdisp, k=args.k, nums=args.nums)
    print("-- model using FastLFNet --")

    if args.cuda:
        model = nn.DataParallel(model)
    model.to(device)

    if args.loadmodel is not None:
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])
        print("-- checkpoint loaded --")
    else:
        print("-- no checkpoint --")

    evaluate_model(args, device, model, TestImgLoader)


def evaluate_model(args, device, model, TestImgLoader):
    # start evaluating
    model.eval()

    ## evaluating ##
    for batch_idx, (center_input,
                    view1_input, view2_input, view3_input, view4_input,
                    class_name) in enumerate(TestImgLoader):
        # --------------------------------------------------
        center_input = center_input.float().to(device)
        view1_input = view1_input.float().to(device)
        view2_input = view2_input.float().to(device)
        view3_input = view3_input.float().to(device)
        view4_input = view4_input.float().to(device)

        print('-------model start, batch: %s-------'%class_name[0])
        start_time = time.time()
        with torch.no_grad():
            output, _ = model(center_input, view1_input, view2_input, view3_input, view4_input)
        use_time = time.time() - start_time
        print('-------model end, batch: %s-------'%class_name[0])

        output = torch.squeeze(output, 1)[0].cpu().numpy()

        if not os.path.exists(args.outpath):
            os.makedirs(args.outpath)

        path_disp_map = os.path.join(args.outpath, 'disp_maps')
        if not os.path.exists(path_disp_map):
            os.makedirs(path_disp_map)

        path_runtimes = os.path.join(args.outpath, 'runtimes')
        if not os.path.exists(path_runtimes):
            os.makedirs(path_runtimes)

        # save .pfm file
        rp.write_pfm(output, os.path.join(path_disp_map, "%s.pfm" %class_name[0]))

        # save predict image
        output_full = 25 * output + 100
        DA.save_disparity_jet(output_full, os.path.join(path_disp_map, "%s.png" %class_name[0]))

        # save running time
        txt_name = os.path.join(path_runtimes, "%s.txt" %class_name[0])
        with open(txt_name, "a") as f:
            f.write(str(use_time))

        print('-------one batch done!!! Used time: %f-------'%use_time)


if __name__ == '__main__':
    cuda_gpu = torch.cuda.is_available()
    if (cuda_gpu):
        print("Great, you have a GPU!")
    else:
        print("Life is short -- consider a GPU!")

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    torch.manual_seed(args.seed)  # Set the seed to generate random numbers
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    evaluate(args, device)

