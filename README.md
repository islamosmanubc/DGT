# Lifelong Learning using a Dynamically Growing Tree of Sub-networks for Out-of-Domain Foreground Segmentation
[![Python](https://img.shields.io/badge/python-3.8-blue.svg?style=flat-square&logo=python&color=3776AB)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.7.0-orange.svg?style=flat-square&logo=tensorflow&color=FF6F00)](https://github.com/y2l/meta-transfer-learning/tree/master/tensorflow)

This repository has the implementation for the paper "Lifelong Learning using a Dynamically Growing Tree of Sub-networks for Out-of-Domain Foreground Segmentation" by Islam Osman and Mohamed S. Shehata

#### Summary
* [Introduction](#introduction)
* [Datasets](#datasets)
* [Visualization](#visualization)
* [Results](#results)
* [Installation](#installation)
* [How to use](#how-to-use)

## Introduction
This paper proposes a lifelong learning technique using dynamically growing tree of sub-networks (DGT). This tree dynamically add sub-networks to learn from new videos. Hence, reduce the catastrophic forgetting problem. Additionally, DGT can learn from new videos using a few number of labeled frames by clustering each new video to the most similar group of videos. To summarize, the novelties of this paper are as follows: 

* DGT achieves the optimal performance for a dataset of videos by clustering the videos into groups of similar videos, and assigning a set of parameters to segment each group. 
    
* DGT solves the problem of catastrophic forgetting as learning to segment new videos will update a set of parameters that is isolated from other set of parameters. Hence, parameters used to segment previously trained on videos will not be affected.
* DGT can learn to segment new videos using a few number of labeled frames by fine-tuning the set of parameters that is used to segment similar videos.

<p align="center">
    <img src="https://github.com/islamosmanubc/DGT/blob/main/figures/test1.png" width="1200"/>
</p>

> Figure: Process of DGT in testing phase. This is the tree generated after training using DAVIS16. The new video is from the testing set and does not exist in the tree. The process of finding the suitable node is done using the greedy algorithm used in lifelong learning phase. The color of network's blocks represent which node is used to generate the parameters.

## Datasets
Densely Annotated VIdeo Segmentation (DAVIS) 2016 and 2017 is a video object segmentation dataset. Each video has a number of frames ranges from 50 to 104. In DAVIS16, a single object is annotated, which is the object of interest in this video. The videos are split into 33 training videos and 20 testing videos. On the other hand, DAVIS17 has multiple objects annotations. The videos are split by frames into training and validation sets.

SegTrackV2 is a small-scale dataset of video multiple objects segmentation. The number of videos in the dataset is 13, and the number of frames ranged from 21 to 279 in each video. The videos are split into 7 training videos and 6 validation videos.

YOutube Video Object Segmentation (YOVOS)
A large-scale dataset that is collected from YouTube video clips. The dataset has 94 different object categories, and annotations for more than 190,000 object. We pre-processed this dataset to fit in the foreground segmentation context. The pre-processing used is assigning one label for all objects in a given video instead of a different label for each object. Then, the dataset is split frame-wise into training and testing sets to be used in the fully supervised experiment. 

ChangeDetection.Net (CDNet) is a benchmark dataset used to detect changes between frames of a video. These changes are the foreground objects. Different challenges are presented in this dataset such as illumination change, camera jitters, night vision, shadows, and dynamic background. The number of videos in each challenge ranges between 4 to 6 videos. The videos are split the same way as YOVOS.

## Visualization
<p align="center">
    <img src="https://github.com/islamosmanubc/DGT/blob/main/figures/full_large_tree_2.5k2.png" width="800"/>
</p>

> Figure: Visualization of DGT-Net_L as circular tree after training using YOVOS, CDNet, and DAVIS17. The node in the middle is the root of the tree, nodes in the first inner circle are children of the root node. The node color defines at which stage the node was added. Blue nodes are added during initial phase using YOVOS. Red nodes are added during lifelong learning phase using CDNet. Finally, black nodes are added during the few-shot learning phase using DAVIS17. The visualization is made using ETE toolkit.

<p align="center">
    <img src="https://github.com/islamosmanubc/DGT/blob/main/figures/visres.png" width="800"/>
</p>

> Figure: Visual results of generalized foreground segmentation of our DGT-Net_L.

<p align="center">
    <img src="https://github.com/islamosmanubc/DGT/blob/main/figures/visres1.png" width="600"/>
</p>

> Figure: Visual results of generalized foreground segmentation of our DGT against other models.

## Results

## Installation

## How to use
