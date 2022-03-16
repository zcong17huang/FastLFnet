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
parser.add_argument('--loadmodel', default='./checkpoints/FastLFnet_model.tar', help='load model')
parser.add_argument('--outpath', default='./outputs/FastLFnet_out', help='output path')
parser.add_argument('--logname', default='log_FastLFnet.log', help='name of the logger')
parser.add_argument('--nums', default=9, type=int, help='nums of NxN')
parser.add_argument('--maxdisp', type=int, default=16, help='maxium disparity')
parser.add_argument('--k', type=int, default=0, help='downsample layers')

parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()

def evaluate(args, device, log):
    im_name_all = lf.Testdataloader(args.datapath)
    im_name_all.sort()

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(im_name_all, args.testsize, args.id_start, args.nums, boolmask_img4=None, boolmask_img6=None, boolmask_img15=None, iftrain=False),
                                    batch_size=args.batchsize, num_workers=8, pin_memory=True, drop_last=True)

    model = FastLFnet(device=device, maxdisp=args.maxdisp, k=args.k, nums=args.nums)
    log.info("-- model using FastLFNet --")

    if args.cuda:
        model = nn.DataParallel(model)
    model.to(device)

    if args.loadmodel is not None:
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])
        log.info("-- checkpoint loaded --")
    else:
        log.info("-- no checkpoint --")

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    evaluate_model(args, device, model, TestImgLoader, log)


def evaluate_model(args, device, model, TestImgLoader, log):
    # start evaluating
    model.eval()
    total_mean_absolute_error = 0
    total_mean_squared_error_x100 = 0
    total_bad_pixel_ratio = 0

    ## evaluating ##
    for batch_idx, (center_input,
                    view1_input, view2_input, view3_input, view4_input,
                    disp_batch_label) in enumerate(TestImgLoader):
        # --------------------------------------------------
        center_input = center_input.float().to(device)
        view1_input = view1_input.float().to(device)
        view2_input = view2_input.float().to(device)
        view3_input = view3_input.float().to(device)
        view4_input = view4_input.float().to(device)

        disp_batch_label = disp_batch_label.float().to(device)

        with torch.no_grad():
            output = model(center_input, view1_input, view2_input, view3_input, view4_input)

        disp_batch_label = disp_batch_label[0].cpu().numpy()
        output = torch.squeeze(output, 1)[0].cpu().numpy()
        assert output.shape[-1] == disp_batch_label.shape[-1]


        diff = np.abs(output - disp_batch_label)
        # save error map
        DA.save_disparity_jet(diff, os.path.join(args.outpath, "%d_error.jpg" % batch_idx))

        bp = (diff >= 0.07)
        mean_absolute_error = np.average(diff)  # end-point-error
        mean_squared_error_x100 = 100 * np.average(np.square(diff))
        bad_pixel_ratio = 100 * np.average(bp)

        if not os.path.exists(args.outpath):
            os.makedirs(args.outpath)

        disp_batch_label = 25 * disp_batch_label + 100  # Scale the disparity range to [0,200]
        output_full = 25 * output + 100
        # save predict image
        DA.save_disparity_jet(disp_batch_label, os.path.join(args.outpath, "%d_gt.jpg" % batch_idx))
        DA.save_disparity_jet(output_full, os.path.join(args.outpath, "%d_out.jpg" % batch_idx))

        log.info('Iter %d MAE_disp = %.5f, MSE_disp_x100 = %.5f, bp_ratio = %.5f' % (batch_idx,
                                                                                     mean_absolute_error,
                                                                                     mean_squared_error_x100,
                                                                                     bad_pixel_ratio))
        total_mean_absolute_error += mean_absolute_error
        total_mean_squared_error_x100 += mean_squared_error_x100
        total_bad_pixel_ratio += bad_pixel_ratio

    average_mean_absolute_error = total_mean_absolute_error / len(TestImgLoader)
    average_mean_squared_error_x100 = total_mean_squared_error_x100 / len(TestImgLoader)
    average_bad_pixel_ratio = total_bad_pixel_ratio / len(TestImgLoader)

    log.info('Average: evaluate MAE_disp = %.5f, evaluate MSE_disp_x100 = %.5f, evaluate bp_ratio = %.5f' % (
                                                                                    average_mean_absolute_error,
                                                                                    average_mean_squared_error_x100,
                                                                                    average_bad_pixel_ratio
                                                                                            ))

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

    log = logger(os.path.join(os.getcwd(), 'evaluate' + args.logname))
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ':' + str(value))

    evaluate(args, device, log)

