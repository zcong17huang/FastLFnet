# FastLFnet: Fast Light-field Disparity Estimation with Multi-disparity-scale Cost Aggregation

**PyTorch implementation of "Fast Light-field Disparity Estimation with Multi-disparity-scale Cost Aggregation", ICCV2021.**

## [Project page](https://computationalperceptionlab.github.io/publications/publications.html) | [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_Fast_Light-Field_Disparity_Estimation_With_Multi-Disparity-Scale_Cost_Aggregation_ICCV_2021_paper.html)

In this paper, we design a lightweight disparity estimation model with physical-based multi-disparity-scale cost volume aggregation for fast disparity estimation. By introducing a sub-network of edge guidance, we significantly improve the recovery of geometric details near edges and improve the overall performance. We significantly reduce computation cost and GPU memory consumption on both densely and sparsely sampled light fields.

![](https://github.com/zcong17huang/FastLFnet/blob/main/configs/net.png)

## Requirement



## Training



## Testing



## Citiation

If you find our code or paper helps, please consider citing:

```
@InProceedings{Huang_2021_ICCV,
    author    = {Huang, Zhicong and Hu, Xuemei and Xue, Zhou and Xu, Weizhu and Yue, Tao},
    title     = {Fast Light-Field Disparity Estimation With Multi-Disparity-Scale Cost Aggregation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {6320-6329}
}
```



## Relevant Works

[LFattNet: Attention-based View Selection Networks for Light-field Disparity Estimation (AAAI 2020)](https://github.com/LIAGM/LFattNet)  
Yu-Ju Tsai, Yu-Lun Liu, Ming Ouhyoung, Yung-Yu Chuang

[EPINET: A Fully-Convolutional Neural Network using Epipolar Geometry for Depth from Light Field Images (CVPR 2018)](https://github.com/chshin10/epinet)  
Changha Shin, Hae-Gon Jeon, Youngjin Yoon, In So Kweon, Seon Joo Kim

[PSMNet: Pyramid Stereo Matching Network (CVPR 2018)](https://github.com/JiaRenChang/PSMNet)  
Jia-Ren Chang, Yong-Sheng Chen
