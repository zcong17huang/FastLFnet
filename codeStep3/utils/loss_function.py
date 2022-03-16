##————————————————————————————##
import torchvision
import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import warnings


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2

    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blured tensors
    """

    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, win, data_range=255, size_average=True, full=False):
    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
    Returns:
        torch.Tensor: ssim results
    """

    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * ( gaussian_filter(X * X, win) - mu1_sq )
    sigma2_sq = compensation * ( gaussian_filter(Y * Y, win) - mu2_sq )
    sigma12   = compensation * ( gaussian_filter(X * Y, win) - mu1_mu2 )

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
    Returns:
        torch.Tensor: ssim results
    """

    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError('Window size should be odd.')

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)

    ssim_val, cs = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=False,
                         full=True)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ms_ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False, weights=None):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        weights (list, optional): weights for different levels
    Returns:
        torch.Tensor: ms-ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if weights is None:
        weights = torch.FloatTensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(X.device, dtype=X.dtype)

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim(X, Y,
                             win=win,
                             data_range=data_range,
                             size_average=False,
                             full=True)
        mcs.append(cs)

        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)
    # weights, (level)
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1))
                            * (ssim_val ** weights[-1]), dim=0)  # (batch, )

    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3):
        r""" class for ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
        """

        super(SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range

    def forward(self, X, Y):
        return ssim(X, Y, win=self.win, data_range=self.data_range, size_average=self.size_average)


class MS_SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3, weights=None):
        r""" class for ms-ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
        """

        super(MS_SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights

    def forward(self, X, Y):
        return ms_ssim(X, Y, win=self.win, size_average=self.size_average, data_range=self.data_range, weights=self.weights)

#VGG loss
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks).cuda()
        self.transform = torch.nn.functional.interpolate
        self.mean =torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).cuda())
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).cuda())
        self.resize = resize

    def forward(self, input, target):

        input = input.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.mean(error)
        return loss

class L1_Gradient_loss(nn.Module):
    def __init__(self):
        super(L1_Gradient_loss, self).__init__()
        self.eps = 1e-6
        self.crit = L1_Charbonnier_loss()

    def forward(self, X, Y):
        xgin = X[:,:,1:,:] - X[:,:,0:-1,:]
        ygin = X[:,:,:,1:] - X[:,:,:,0:-1]
        xgtarget = Y[:,:,1:,:] - Y[:,:,0:-1,:]
        ygtarget = Y[:,:,:,1:] - Y[:,:,:,0:-1]

        xl = self.crit(xgin, xgtarget)
        yl = self.crit(ygin, ygtarget)
        return (xl + yl) * 0.5

class SemanticBoundaryLoss(nn.Module):
    def __init__(self, device, reduction = 'mean'):
        super(SemanticBoundaryLoss, self).__init__()
        self.device = device
        self.reduction = reduction

    def forward(self, sem_images, gray_images: torch.Tensor) -> torch.Tensor:
        channels = sem_images.shape[1]
        if channels != 1:
            warnings.warn('The channel of input is not 1, suggest using gray images to get gradient!!!')

        gradient_tensor_x = torch.Tensor([[1.0, 0.0, -1.0],
                                          [2.0, 0.0, -2.0],
                                          [1.0, 0.0, -1.0]]).to(self.device).view((1, 1, 3, 3))
        gradient_tensor_y = torch.Tensor([[1.0, 2.0, 1.0],
                                          [0.0, 0.0, 0.0],
                                          [-1.0, -2.0, -1.0]]).to(self.device).view((1, 1, 3, 3))

        Sem_x = F.conv2d(sem_images, gradient_tensor_x, padding=1)
        Gray_x = F.conv2d(gray_images, gradient_tensor_x, padding=1)

        Sem_y = F.conv2d(sem_images, gradient_tensor_y, padding=1)
        Gray_y = F.conv2d(gray_images, gradient_tensor_y, padding=1)

        ret = torch.abs(Sem_x)*torch.exp(-torch.abs(Gray_x)) + torch.abs(Sem_y)*torch.exp(-torch.abs(Gray_y))

        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)

        return ret

class DispEdgeLoss(nn.Module):
    def __init__(self, device):
        super(DispEdgeLoss, self).__init__()
        self.device = device
        self.loss = nn.L1Loss()
    def forward(self, gt_disp, out_edge):
        gt_disp = gt_disp / 175. # normalized to [0,1]

        channels = gt_disp.shape[1]
        if channels != 1:
            warnings.warn('The channel of input is not 1, suggest using gray images to get gradient!!!')

        gradient_tensor_x = torch.tensor([[1.0, 0.0, -1.0],
                                          [2.0, 0.0, -2.0],
                                          [1.0, 0.0, -1.0]]).to(self.device).view((1, 1, 3, 3))
        gradient_tensor_y = torch.tensor([[1.0, 2.0, 1.0],
                                          [0.0, 0.0, 0.0],
                                          [-1.0, -2.0, -1.0]]).to(self.device).view((1, 1, 3, 3))

        Edge_x = F.conv2d(gt_disp, gradient_tensor_x, padding=1)
        Edge_y = F.conv2d(gt_disp, gradient_tensor_y, padding=1)

        gt_edge = torch.sqrt(torch.pow(Edge_x, 2) + torch.pow(Edge_y, 2))
        gt_edge = gt_edge / 4.

        gt_edge = gt_edge * 255.    # normalized to [0,255]
        out_edge = out_edge *255.
        return self.loss(gt_edge, out_edge)


class GradientLoss(nn.Module):
    def __init__(self, device):
        super(GradientLoss, self).__init__()
        self.device = device
        self.loss = nn.SmoothL1Loss()
    def forward(self, gt_disp, out_disp):
        gt_disp = gt_disp * 255. / 175.
        out_disp = out_disp * 255. / 175.   # normalized to [0,255]

        channels = gt_disp.shape[1]
        if channels != 1:
            warnings.warn('The channel of input is not 1, suggest using gray images to get gradient!!!')

        gradient_tensor_x = torch.tensor([[1.0, 0.0, -1.0],
                                          [2.0, 0.0, -2.0],
                                          [1.0, 0.0, -1.0]]).to(self.device).view((1, 1, 3, 3))
        gradient_tensor_y = torch.tensor([[1.0, 2.0, 1.0],
                                          [0.0, 0.0, 0.0],
                                          [-1.0, -2.0, -1.0]]).to(self.device).view((1, 1, 3, 3))

        gt_Edge_x = F.conv2d(gt_disp, gradient_tensor_x, padding=1)
        gt_Edge_y = F.conv2d(gt_disp, gradient_tensor_y, padding=1)
        gt_edge = torch.sqrt(torch.pow(gt_Edge_x, 2) + torch.pow(gt_Edge_y, 2) + 1e-10)
        gt_edge = gt_edge / 4.

        out_Edge_x = F.conv2d(out_disp, gradient_tensor_x, padding=1)
        out_Edge_y = F.conv2d(out_disp, gradient_tensor_y, padding=1)
        out_edge = torch.sqrt(torch.pow(out_Edge_x, 2) + torch.pow(out_Edge_y, 2) + 1e-10)
        out_edge = out_edge / 4.

        return self.loss(gt_edge, out_edge)


class ImageGradientLoss(_Loss):
    def __init__(self, device, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(ImageGradientLoss, self).__init__(size_average, reduce, reduction)
        self.device = device

    def forward(self, logits: torch.Tensor, gray_images: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)

        gradient_tensor_x = torch.Tensor([[1.0, 0.0, -1.0],
                                          [2.0, 0.0, -2.0],
                                          [1.0, 0.0, -1.0]]).to(self.device).view((1, 1, 3, 3))
        gradient_tensor_y = torch.Tensor([[1.0, 2.0, 1.0],
                                          [0.0, 0.0, 0.0],
                                          [-1.0, -2.0, -1.0]]).to(self.device).view((1, 1, 3, 3))

        I_x = F.conv2d(gray_images, gradient_tensor_x)
        M_x = F.conv2d(pred, gradient_tensor_x)

        I_y = F.conv2d(gray_images, gradient_tensor_y)
        M_y = F.conv2d(pred, gradient_tensor_y)

        G = torch.sqrt(torch.pow(M_x, 2) + torch.pow(M_y, 2))

        gradient = 1 - torch.pow(I_x * M_x + I_y * M_y, 2)

        image_gradient_loss = torch.sum(torch.mul(G, gradient)) / (torch.sum(G) + 1e-6)

        image_gradient_loss = image_gradient_loss if image_gradient_loss > 0 else 0

        return image_gradient_loss


class HairMattingLoss(_Loss):
    def __init__(self,
                 device,
                 gradient_loss_weight: float = 0.5,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean') -> None:
        super(HairMattingLoss, self).__init__(size_average, reduce, reduction)
        self.device = device
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.gradient_loss = ImageGradientLoss(device)
        self.gradient_loss_weight = gradient_loss_weight

    def forward(self, logits: torch.Tensor, masks: torch.Tensor, gray_images: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce_loss(logits, masks)
        gradient_loss = self.gradient_loss(logits, gray_images)
        loss = bce_loss + self.gradient_loss_weight * gradient_loss
        return loss