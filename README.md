# Lifelong Learning using a Dynamically Growing Tree of Sub-networks for Out-of-Domain Foreground Segmentation
[![Python](https://img.shields.io/badge/python-3.8-blue.svg?style=flat-square&logo=python&color=3776AB)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.7.0-orange.svg?style=flat-square&logo=tensorflow&color=FF6F00)](https://github.com/y2l/meta-transfer-learning/tree/master/tensorflow)

This repository has the implementation for the paper "Lifelong Learning Using a Dynamically Growing Tree of Sub-networks for Domain Generalization in Video Object Segmentation" by Islam Osman and Mohamed S. Shehata.

#### Summary
* [Introduction](#introduction)
* [Datasets](#datasets)
* [Visualization](#visualization)
* [Results](#results)
* [Installation](#installation)
* [Pre-trained Weights](#pre-trained)

## Introduction
This paper proposes a lifelong learning technique using a dynamically growing tree of sub-networks (DGT). This tree dynamically adds sub-networks to learn from new videos. Hence, reduces the catastrophic forgetting problem. Additionally, DGT can learn from new videos using a few labeled frames by clustering each new video to the most similar group of videos. To summarize, the novelties of this paper are as follows: 

* DGT achieves the optimal performance for a dataset of videos by clustering the videos into groups of similar videos and assigning a set of parameters to segment each group. 
    
* DGT solves the problem of catastrophic forgetting as learning to segment new videos will update a set of parameters isolated from other sets. Hence, parameters used to segment previously trained videos will not be affected.
* DGT can learn to segment new videos using a few labeled frames by fine-tuning the set of parameters used to segment similar videos.
<p align="center">
    <img src="https://github.com/islamosmanubc/DGT/blob/main/figures/dgtpro.png" width="1200"/>
</p>

> Figure: Process of DGT in the testing phase. First, an agent requests a suitable network from DGT for the new video using a labeled reference frame. Then, DGT generates a task-specific network by selecting a suitable node for each layer of the network. Finally, the generated network is sent to the agent to segment the new video frames.

## Datasets
Densely Annotated VIdeo Segmentation (DAVIS) 2016 and 2017 is a video object segmentation dataset. Each video has a number of frames ranging from 50 to 104. In DAVIS16, a single object is annotated, which is the object of interest in this video. The videos are split into 33 training videos and 20 testing videos. On the other hand, DAVIS17 has multiple object annotations. The videos are split by frames into training and validation sets.

SegTrackV2 is a small-scale dataset of video multiple objects segmentation. The number of videos in the dataset is 13, and the number of frames ranged from 21 to 279 in each video. The videos are split into seven training videos and six validation videos.

YouTube Video Object Segmentation (YT-VOS18)
A large-scale dataset that is collected from YouTube video clips. The dataset has 94 different object categories and annotations for more than 190,000 objects. 

ChangeDetection.Net (CDNet) is a benchmark dataset used to detect changes between video frames. These changes are the foreground objects. Different challenges are presented in this dataset, such as illumination change, camera jitters, night vision, shadows, and dynamic background. The number of videos in each challenge ranges between 4 to 6 videos. The videos are split the same way as YOVOS.

## Visualization
<p align="center">
    <img src="https://github.com/islamosmanubc/DGT/blob/main/figures/full_large_tree_2.5k2.png" width="800"/>
</p>

> Figure: Visualization of DGT-Net_L as a circular tree after training using YT-VOS18, CDNet, and DAVIS17. The node in the middle is the root of the tree, and the nodes in the first inner circle are children of the root node. The node color defines at which stage the node was added. Blue nodes are added during the initial phase using YT-VOS18. Red nodes are added during the lifelong learning phase using CDNet. Finally, black nodes are added during the few-shot learning phase using DAVIS17. The visualization is made using the ETE toolkit.

<p align="center">
    <img src="https://github.com/islamosmanubc/DGT/blob/main/figures/mres.png" width="800"/>
</p>

> Figure: Visual results of multiple source domains for video object segmentation of our DGT-Net_L.

<p align="center">
    <img src="https://github.com/islamosmanubc/DGT/blob/main/figures/visres1.png" width="600"/>
</p>

> Figure: Visual results of multiple source domains for video object segmentation of our DGT against other models.

## Results
|          Methods       |       DAVIS16     |      DAVIS17     |    YOVOS    | Cont. CDNet  |Cont. SegTrackV2 1-shot| 
| ---------------------- | ----------------- | ---------------- | ----------- | ------------ | --------------------- |
| `DGT-NetS`             |      `0.940`      |      `0.921`     |   `0.832`   |   `0.689`    |         `0.836`       |
| `DGT-NetL`             |      `0.944`      |      `0.932`     |   `0.856`   |   `0.733`    |         `0.842`       |


## Installation
To run the code, you need to install:
* python 3.8
* Tensorflow-gpu v2.7.0

Also, you need to install some extra packages:
* scipy
* scikit
* pillow

## Pre-trained Weights
The pre-trained weights are available on this link:
[pre-trained weights](https://drive.google.com/drive/folders/1PIw4zsekFuQqzrDf5NDJ_yLReTqu8xIq?usp=sharing)

The "base_weights" folder has the weights of the trained model without DGT. On the other hand, "growing_tree_weights" has the weights of the full tree DGT.
There are three files: 1 for davis16, 1 for segtrackv2, and 1 for the DGT generalization experiment shown in the paper that uses all three datasets, namely, YT-VOS18, CDNet, and DAVIS17.
