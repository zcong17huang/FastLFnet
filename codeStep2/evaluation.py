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
from utils.logger import setup_logger1 as logger
import utils.readpfm as rp
from dataloader import listLFfile as lf
from dataloader import HCILoader as DA
from models.FastLFNet import FastLF_Net as FastLFnet

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='FastLFNet')
parser.add_argument("--batchsize", type=int, default=1)
parser.add_argument('--testsize', type=int, default=512, help='test patch size as input of net')
parser.add_argument('--id_start', type=int, default=0, help='start index of test patch')
parser.add_argument('--datapath', default='/data1/Dataset/HCI_Benchmark/', help='datapath')
parser.add_argument('--loadmodel', default='./checkpoints/FastLFnet_model_edge.tar', help='load model')
parser.add_argument('--outpath', default='./outputs/FastLFnet_out', help='output path')
parser.add_argument('--nums', default=9, type=int, help='nums of NxN')
parser.add_argument('--maxdisp', type=int, default=16, help='maxium disparity')
parser.add_argument('--k', type=int, default=0, help='downsample layers')

parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()

def evaluate(args, device):
    im_name_all = lf.Testdataloader(args.datapath)
    im_name_all.sort()

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(im_name_all, args.testsize, args.id_start, args.nums, iftrain=False),
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

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    evaluate_model(args, device, model, TestImgLoader)


def evaluate_model(args, device, model, TestImgLoader):
    # start evaluating
    model.eval()

    ## testing ##
    for batch_idx, (center_input, edge_binary_label) in enumerate(TestImgLoader):
        # --------------------------------------------------
        center_input = center_input.float().to(device)

        with torch.no_grad():
            edge_out = model(center_input)

        if not os.path.exists(args.outpath):
            os.makedirs(args.outpath)

        # save center image
        center_input = torch.squeeze(center_input, 1)[0].cpu().numpy()
        center_input = np.clip(center_input, 0.0, 1.0)
        center_input = (center_input * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(args.outpath, "%d_center.png" % batch_idx), center_input)

        # save edge map
        edge_out = torch.squeeze(edge_out, 1)[0].cpu().numpy()
        edge_out = np.clip(edge_out, 0.0, 1.0)
        edge_out = (edge_out * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(args.outpath, "%d_edge.png" % batch_idx), edge_out)

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

