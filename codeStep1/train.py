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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='FastLFNet')
parser.add_argument("--train_batchsize", type=int, default=8)
parser.add_argument('--trainsize', type=int, default=32, help='train patch size as input of net')
parser.add_argument("--test_batchsize", type=int, default=1)
parser.add_argument('--testsize', type=int, default=512, help='test patch size as input of net')
parser.add_argument('--id_start', type=int, default=0, help='start index of test patch')
parser.add_argument("--learning_rate", default=1e-3)
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
parser.add_argument('--test_epoch', default=700, help='test epoch start')
parser.add_argument('--test_freq', type=int, default=10, help='test and save model frequence')
parser.add_argument('--trainlist_cycles', type=int, default=20, help='circulations of train list, 1 epoch is equal to 4*cycles epochs now')
parser.add_argument('--datapath', default='/data1/Dataset/HCI_Benchmark/', help='datapath')
parser.add_argument('--loadmodel', default=None, help='load model')
parser.add_argument('--savemodel', default='./checkpoints/FastLFnet_model.tar', help='save model')
parser.add_argument('--logname', default='log_FastLFnet.log', help='name of the logger')
parser.add_argument('--nums', default=9, type=int, help='nums of NxN')
parser.add_argument('--steps', default=[250,500,750], help='learning rate step size')
parser.add_argument('--gamma', '--gm', default=0.5, type=float, help='learning rate decay parameter: Gamma')
parser.add_argument('--maxdisp', type=int, default=16, help='maxium disparity')
parser.add_argument('--k', type=int, default=0, help='downsample layers')

parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()


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
    log_test = logger2(os.path.join(os.getcwd(), 'test' + args.logname))

    for key, value in sorted(vars(args).items()):
        log_train.info(str(key) + ':' + str(value))
    # ----------------------------------------------------------------------
    '''load invalid regions from training data (ex. reflective region)'''
    boolmask_img4 = imageio.imread(os.path.normcase(os.path.join(args.datapath, 'full_data/additional_invalid_area/kitchen/input_Cam040_invalid_ver2.png')))
    boolmask_img6 = imageio.imread(os.path.normcase(os.path.join(args.datapath, 'full_data/additional_invalid_area/kitchen/input_Cam040_invalid_ver2.png')))
    boolmask_img15 = imageio.imread(os.path.normcase(os.path.join(args.datapath, 'full_data/additional_invalid_area/kitchen/input_Cam040_invalid_ver2.png')))
    boolmask_img4 = 1.0 * boolmask_img4[:, :, 3] > 0
    boolmask_img6 = 1.0 * boolmask_img6[:, :, 3] > 0
    boolmask_img15 = 1.0 * boolmask_img15[:, :, 3] > 0

    train_im_name_all = lf.Traindataloader(args.datapath)
    train_im_name_all_extend =[]
    for i in range(args.trainlist_cycles):
        random.shuffle(train_im_name_all)
        train_im_name_all_extend.extend(train_im_name_all)

    test_im_name_all = lf.Testdataloader(args.datapath)
    test_im_name_all.sort()

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_im_name_all_extend, args.trainsize, args.id_start, args.nums, boolmask_img4, boolmask_img6, boolmask_img15, iftrain=True),
                                        batch_size=args.train_batchsize, num_workers=8, pin_memory=True, drop_last=True)
    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_im_name_all, args.testsize, args.id_start, args.nums, boolmask_img4, boolmask_img6, boolmask_img15, iftrain=False),
                                        batch_size=args.test_batchsize, num_workers=8, pin_memory=True, drop_last=True)

    model = FastLFnet(device=device, maxdisp=args.maxdisp, k=args.k, nums=args.nums)
    log_train.info("-- model using FastLFNet --")

    if args.cuda:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    epoch_start = 0
    last_loss = float('inf')

    if args.loadmodel is not None:
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        epoch_start = state_dict['epoch']
        last_loss = state_dict['loss_best']
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps,
                                                    gamma=args.gamma, last_epoch=epoch_start)
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
    length_test = len(TestImgLoader)
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

        if epoch < args.test_epoch:
            if epoch % args.test_freq == 0:
                # Test
                test_MAE, test_MSE_x100, test_bp = test_model(device, model, TestImgLoader, length_test, log_test)
                writer_name = 'test' + args.logname
                writer.add_scalar(writer_name, test_MSE_x100, epoch)
                writer.close()
                # SAVE
                if test_MSE_x100 < last_loss:
                    last_loss = test_MSE_x100
                    torch.save({
                        'state_dict': model.state_dict(),
                        'loss_best': last_loss,
                        'epoch': epoch + 1,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, args.savemodel)

        if epoch >= args.test_epoch:
            # Test
            test_MAE, test_MSE_x100, test_bp = test_model(device, model, TestImgLoader,
                                                          length_test, log_test, if_info_Iter=True)
            writer_name = 'test' + args.logname
            writer.add_scalar(writer_name, test_MSE_x100, epoch)
            writer.close()
            # SAVE
            if test_MSE_x100 < last_loss:
                last_loss = test_MSE_x100
                torch.save({
                    'state_dict': model.state_dict(),
                    'loss_best': last_loss,
                    'epoch': epoch + 1,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, args.savemodel)

    # SAVE_LAST
    last_save = os.path.splitext(args.savemodel)[0] + '_last'+ os.path.splitext(args.savemodel)[1]
    torch.save({
        'state_dict': model.state_dict(),
        'loss_best': last_loss,
        'epoch': args.epochs,
        'optimizer_state_dict': optimizer.state_dict(),
    }, last_save)

    log_train.info('full training time = %.4f Hours' % ((time.time() - start_full_time) / 3600))


def train_model(device, model, scheduler, TrainImgLoader, optimizer, length_train):
    # start training
    model.train()
    total_train_loss = 0

    ## training ##
    for batch_idx, (center_input,
                    view1_input, view2_input, view3_input, view4_input,
                    disp_batch_label) in enumerate(TrainImgLoader):
        # --------------------------------------------------
        optimizer.zero_grad()

        center_input = center_input.float().to(device)
        view1_input = view1_input.float().to(device)
        view2_input = view2_input.float().to(device)
        view3_input = view3_input.float().to(device)
        view4_input = view4_input.float().to(device)

        disp_batch_label = torch.unsqueeze(disp_batch_label.float().to(device), 1)

        output = model(center_input, view1_input, view2_input, view3_input, view4_input)

        output = 25 * output + 100
        disp_batch_label = 25 * disp_batch_label + 100                # Scale the disparity range to [0,200]

        loss1 = nn.SmoothL1Loss()
        # loss2 = losses.SSIM(data_range=32., channel=1).to(device)
        # loss3 = losses.SemanticBoundaryLoss(device)
        # loss4 = nn.L1Loss()

        loss = loss1(disp_batch_label, output)

        loss.backward()
        optimizer.step()
        loss = loss.item()
        # -----------------------------------------------
        total_train_loss += loss

    scheduler.step()
    average_loss = total_train_loss/length_train

    return average_loss


def test_model(device, model, TestImgLoader, length_test, log, if_info_Iter=False):
    # start evaluating
    model.eval()
    total_mean_absolute_error = 0
    total_mean_squared_error_x100 = 0
    total_bad_pixel_ratio = 0

    ## testing ##
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
        bp = (diff >= 0.07)
        mean_absolute_error = np.average(diff)  # end-point-error
        mean_squared_error_x100 = 100 * np.average(np.square(diff))
        bad_pixel_ratio = 100 * np.average(bp)

        if if_info_Iter == True:
            log.info('Iter %d MAE_disp = %.5f, MSE_disp_x100 = %.5f, bp_ratio = %.5f' % (batch_idx,
                                                                                         mean_absolute_error,
                                                                                         mean_squared_error_x100,
                                                                                         bad_pixel_ratio))
        total_mean_absolute_error += mean_absolute_error
        total_mean_squared_error_x100 += mean_squared_error_x100
        total_bad_pixel_ratio += bad_pixel_ratio

    average_mean_absolute_error = total_mean_absolute_error / length_test
    average_mean_squared_error_x100 = total_mean_squared_error_x100 / length_test
    average_bad_pixel_ratio = total_bad_pixel_ratio / length_test

    log.info('Average: test MAE_disp = %.5f, test MSE_disp_x100 = %.5f, test bp_ratio = %.5f' % (
                                                                                    average_mean_absolute_error,
                                                                                    average_mean_squared_error_x100,
                                                                                    average_bad_pixel_ratio
                                                                                            ))
    return average_mean_absolute_error, average_mean_squared_error_x100, average_bad_pixel_ratio

if __name__ == '__main__':
    main()

