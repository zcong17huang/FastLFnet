# -*- coding: UTF-8 -*-
import argparse
import os
import numpy as np
import imageio
import time
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision
import torch.nn.functional as F
from torch.optim import Adam
from utils import loss_function as losses
from utils.logger import setup_logger1 as logger1
from utils.logger import setup_logger2 as logger2
from dataloader import listLFfile as lf
from dataloader import HCILoader as DA
from models.FastLFNet import FastLF_Net as FastLFnet
from tensorboardX import SummaryWriter
import re
import cv2

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='FastLFNet')
parser.add_argument("--train_batchsize", type=int, default=16)
parser.add_argument('--trainsize', type=int, default=128, help='train patch size as input of net')
parser.add_argument("--test_batchsize", type=int, default=1)
parser.add_argument('--testsize', type=int, default=512, help='test patch size as input of net')
parser.add_argument('--id_start', type=int, default=0, help='start index of test patch')
parser.add_argument("--learning_rate", default=1e-3)
parser.add_argument('--epochs', type=int, default=900, help='number of epochs to train')
parser.add_argument('--test_freq', type=int, default=50, help='test and save model frequence')
parser.add_argument('--trainlist_cycles', type=int, default=30, help='circulations of train list, 1 epoch is equal to 4*cycles epochs now')
parser.add_argument('--datapath', default='/data1/Dataset/HCI_Benchmark/', help='datapath')
parser.add_argument('--loadmodel', default='./checkpoints/FastLFnet_model.tar', help='load model')
parser.add_argument('--savemodel', default='./checkpoints/FastLFnet_model_edge.tar', help='save model')
parser.add_argument('--outpath', default='./outputs', help='output path')
parser.add_argument('--logname', default='log_FastLFnet.log', help='name of the logger')
parser.add_argument('--nums', default=9, type=int, help='nums of NxN')
parser.add_argument('--steps', default=[225,450,675], help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.2, type=float, help='learning rate decay parameter: Gamma')
parser.add_argument('--maxdisp', type=int, default=16, help='maxium disparity')
parser.add_argument('--k', type=int, default=0, help='downsample layers')

parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()

def cross_entropy_loss2d(inputs, targets, device, balance=1.1):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()
    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * balance / valid
    weights = torch.from_numpy(weights).float()
    weights = weights.to(device)

    loss = nn.BCELoss(weights)(inputs, targets)
    return loss

def main():
    # ----------------------------------------------------------------------
    cuda_gpu = torch.cuda.is_available()
    if (cuda_gpu):
        print("Great, you have a GPU!")
    else:
        print("Life is short -- consider a GPU!")

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)  # Set the seed to generate random numbers
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    writer = SummaryWriter()
    log_train = logger1(os.path.join(os.getcwd(), 'train' + args.logname))

    for key, value in sorted(vars(args).items()):
        log_train.info(str(key) + ':' + str(value))
    # ----------------------------------------------------------------------
    train_im_name = lf.Traindataloader(args.datapath)
    random.shuffle(train_im_name)

    test_im_name = lf.Testdataloader(args.datapath)
    test_im_name = test_im_name
    random.shuffle(test_im_name)

    train_im_name_all = train_im_name + test_im_name
    train_im_name_all_extend =[]
    for i in range(args.trainlist_cycles):
        random.shuffle(train_im_name_all)
        train_im_name_all_extend.extend(train_im_name_all)

    test_im_name_all = lf.Testdataloader(args.datapath)
    test_im_name_all.sort()

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_im_name_all_extend, args.trainsize, args.id_start, args.nums, iftrain=True),
                                        batch_size=args.train_batchsize, num_workers=8, pin_memory=True, drop_last=True)
    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_im_name_all, args.testsize, args.id_start, args.nums, iftrain=False),
                                        batch_size=args.test_batchsize, num_workers=8, pin_memory=True, drop_last=True)

    model = FastLFnet(device=device, maxdisp=args.maxdisp, k=args.k, nums=args.nums)
    log_train.info("-- model using FastLFNet --")

    if args.cuda:
        model = nn.DataParallel(model)
    model.to(device)

    base_lr = args.learning_rate
    params = []
    for key, v in model.named_parameters():
        if re.match(r'module.feature_extraction.egde_sub', key):
            params += [{'params': v, 'lr': base_lr * 1., 'name': key}]
        else:
            params += [{'params': v, 'lr': base_lr * 0.0001, 'name': key}]

    optimizer = Adam(params, lr=args.learning_rate)

    epoch_start = 0

    if args.loadmodel is not None:
        model_dict = model.state_dict()

        state_dict = torch.load(args.loadmodel)
        pretrained_dict = state_dict['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.gamma)
        log_train.info("=> loaded checkpoint '{}' (epoch {})".format(args.loadmodel, epoch_start))
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.gamma)
        log_train.info("None loadmodel => will start from scratch.")

    log_train.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    savepath = os.path.dirname(args.savemodel)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
#----------------------------------------------------------------------
    length_train = len(TrainImgLoader)
    start_full_time = time.time()

    for epoch in range(epoch_start, args.epochs):
        # Train
        log_train.info('This is %d-th epoch, learning rate : %f ' % (epoch, scheduler.get_lr()[0]))
        start_time = time.time()

        train_loss = train_model(device, model, scheduler, TrainImgLoader, optimizer, length_train)
        log_train.info('epoch %d total training loss = %.5f, time = %.4f Hours'
                                                    % (epoch, train_loss, (time.time() - start_time) / 3600))
        writer_name = 'train' + args.logname
        writer.add_scalar(writer_name, train_loss, epoch)
        writer.close()

        if epoch % args.test_freq == 0:
            # Test
            test_model(device, model, TestImgLoader, epoch)

            # SAVE
            save_name = os.path.splitext(args.savemodel)[0] + '_epoch%d'%epoch + os.path.splitext(args.savemodel)[1]
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_name)

    # SAVE_LAST
    torch.save({
        'state_dict': model.state_dict(),
        'epoch': args.epochs,
        'optimizer_state_dict': optimizer.state_dict(),
    }, args.savemodel)

    log_train.info('full training time = %.4f Hours' % ((time.time() - start_full_time) / 3600))


def train_model(device, model, scheduler, TrainImgLoader, optimizer, length_train):
    # start training
    model.train()
    total_train_loss = 0

    ## training ##
    for batch_idx, (center_input, edge_binary_label) in enumerate(TrainImgLoader):
        # --------------------------------------------------
        optimizer.zero_grad()

        center_input = center_input.float().to(device)
        edge_binary_label = torch.unsqueeze(edge_binary_label.float().to(device), 1)

        edge_out = model(center_input)

        loss = cross_entropy_loss2d(edge_out, edge_binary_label, device)

        loss.backward()
        optimizer.step()
        loss = loss.item()
        # -----------------------------------------------
        total_train_loss += loss

    scheduler.step()
    average_loss = total_train_loss/length_train

    return average_loss


def test_model(device, model, TestImgLoader, epoch):
    # start evaluating
    model.eval()

    ## testing ##
    for batch_idx, (center_input, edge_binary_label) in enumerate(TestImgLoader):
        # --------------------------------------------------
        center_input = center_input.float().to(device)

        with torch.no_grad():
            edge_out = model(center_input)

        # save edge map
        edge_out = torch.squeeze(edge_out, 1)[0].cpu().numpy()
        edge_out = np.clip(edge_out, 0.0, 1.0)
        edge_out = (edge_out * 255.0).astype(np.uint8)
        save_path = os.path.join(args.outpath, 'epoch_%s'%epoch)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, "%d_edge.png" % batch_idx), edge_out)

if __name__ == '__main__':
    main()

