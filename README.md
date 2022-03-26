# FastLFnet: Fast Light-field Disparity Estimation with Multi-disparity-scale Cost Aggregation

**PyTorch implementation of "Fast Light-field Disparity Estimation with Multi-disparity-scale Cost Aggregation", ICCV2021.**

## [Project page](https://computationalperceptionlab.github.io/publications/publications.html) | [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Huang_Fast_Light-Field_Disparity_Estimation_With_Multi-Disparity-Scale_Cost_Aggregation_ICCV_2021_paper.html)

In this paper, we design a lightweight disparity estimation model with physical-based multi-disparity-scale cost volume aggregation for fast disparity estimation. By introducing a sub-network of edge guidance, we significantly improve the recovery of geometric details near edges and improve the overall performance. We significantly reduce computation cost and GPU memory consumption on both densely and sparsely sampled light fields.

![](https://github.com/zcong17huang/FastLFnet/blob/main/configs/net.png)

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

## Requirement

The code is tested in the following environment. The newer version of the packages should also be fine. And it is highly recommended to use Anaconda.

```
python==3.5.6
tensorboard==1.14.0
torch==1.2.0
torchvision==0.4.0
imageio==2.8.0
keras==2.2.5
matplotlib==3.0.1
numpy==1.18.5
opencv-python==4.1.1.26
pillow==6.2.0
```

## Dataset

###### [4D light field dataset](https://lightfield-analysis.uni-konstanz.de/)

A synthetic dataset with 28 carefully designed scenes, which is partitioned into four subsets: *Stratified*, *Test*, *Training*, and *Additional*. We use the subset of *Additional* for training and the others for validation and testing.

## Training, Evaluating and Submitting

As mentioned in the paper, we use a three-step training strategy to train the network.

- **Step1: Coarse training**

First, we train our FastLFnet without edge guidance to get coarse results.

Go to the folder **[codeStep1](https://github.com/zcong17huang/FastLFnet/tree/main/codeStep1)**, and use the following command to get coarse results and pretrained model.

```
python train.py --epochs 1000 \
                --datapath <your-dataset-folder> \
                --loadmodel <pretrained model, default None> \
                --savemodel <path for saving model> \
                --nums 9 \                
                --maxdisp 16 \
                --no_cuda (if for CPU training)          
```

After training, you can get evaluating results using the following command.

```
python evaluation.py --datapath <your-dataset-folder> \
                     --loadmodel <pretrained model after training> \
                     --outpath <path for saving output results> \
                     --nums 9 \                
                     --maxdisp 16 \
                     --no_cuda (if for CPU evaluating)
```

- **Step2: Edge guidance**

Secondly, we combine the edge guidance sub-network with the feature extraction module to predict the edge information of the center view.

Go to the folder **[codeStep2](https://github.com/zcong17huang/FastLFnet/tree/main/codeStep2)**, and use the following command to train the edge guidance sub-network. *Here we need the pretrained model from Step1*.

```
python train.py --train_batchsize 16 \
                --trainsize 128 \
                --epochs 900 \
                --datapath <your-dataset-folder> \
                --loadmodel <pretrained model from Step1> \
                --savemodel <path for saving model> \
                --no_cuda (if for CPU training) 
```

After training, you can obtain the edge prediction results using the following command.

```
python evaluation.py --datapath <your-dataset-folder> \
                     --loadmodel <pretrained model after training> \
                     --outpath <path for saving output results> \
                     --no_cuda (if for CPU evaluating)
```

- **Step3: Jointly training**

Finally, we train the whole FastLFnet jointly.

Go to the folder **[codeStep3](https://github.com/zcong17huang/FastLFnet/tree/main/codeStep3)**, and use the following command to train the whole network. *Here we need the pretrained model from Step1 and Step2*.

```
python train.py --epochs 1000 \
                --datapath <your-dataset-folder> \
                --loadmodel_1 <pretrained model from Step1> \
                --loadmodel_2 <pretrained model from Step2> \
                --savemodel <path for saving model> \
                --nums 9 \                
                --maxdisp 16 \
                --no_cuda (if for CPU training) 
```

You can get final evaluating results using the following command.

```
python evaluation.py --datapath <your-dataset-folder> \
                     --loadmodel <pretrained model after training> \
                     --outpath <path for saving output results> \
                     --nums 9 \                
                     --maxdisp 16 \
                     --no_cuda (if for CPU evaluating)
```

You can get submission results for the **benchmark** [4D Light Field Dataset](https://lightfield-analysis.uni-konstanz.de/benchmark/table?column-type=images&metric=mse_100) using the following command.

```
python submission512.py --datapath <your-dataset-folder> \
                        --loadmodel <pretrained model after training> \
                        --outpath <path for saving output results> \
                        --nums 9 \                
                        --maxdisp 16 \
                        --no_cuda (if for CPU submitting)
```

## Pretrained Model

Pretrained Model for 4D Light Field Dataset [Google Drive](https://drive.google.com/file/d/1X4CMv2tYt89uxAPtIY5kjl-5eVDMFfih/view?usp=sharing)

## Edge Maps

The [edge maps](https://github.com/zcong17huang/FastLFnet/tree/main/edgeMaps) have been uploaded. Each light field scene contains a corresponding edge map, and you can begin training after placing the edge map in the corresponding scene folder.

## Benchmark submission Results

The submission Results for [4D Light Field Dataset](https://lightfield-analysis.uni-konstanz.de/benchmark/table?column-type=images&metric=mse_100) benchmark are stored in the **[subResults](https://github.com/zcong17huang/FastLFnet/tree/main/subResults/hci)** folder

## Results

- **Comparison in performance and efficiency.**

<img src="https://github.com/zcong17huang/FastLFnet/blob/main/configs/table1.png" width="700" height="557" alt="table1"/><br/>

- **Quantitative comparison on the 4D Light Field Dataset.**

<img src="https://github.com/zcong17huang/FastLFnet/blob/main/configs/table2.png" width="800" height="442" alt="table2"/><br/>

- **Qualitative results**

![](https://github.com/zcong17huang/FastLFnet/blob/main/configs/fig.png)

## Relevant Works

[LFattNet: Attention-based View Selection Networks for Light-field Disparity Estimation (AAAI 2020)](https://github.com/LIAGM/LFattNet)  
Yu-Ju Tsai, Yu-Lun Liu, Ming Ouhyoung, Yung-Yu Chuang

[EPINET: A Fully-Convolutional Neural Network using Epipolar Geometry for Depth from Light Field Images (CVPR 2018)](https://github.com/chshin10/epinet)  
Changha Shin, Hae-Gon Jeon, Youngjin Yoon, In So Kweon, Seon Joo Kim

[PSMNet: Pyramid Stereo Matching Network (CVPR 2018)](https://github.com/JiaRenChang/PSMNet)  
Jia-Ren Chang, Yong-Sheng Chen
