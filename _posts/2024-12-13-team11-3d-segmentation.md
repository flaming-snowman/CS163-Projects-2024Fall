---
layout: post
comments: true
title: 3D Semantic Segmentation
author: Rathul Anand, Jason Liu
date: 2024-12-12
---

> In 3D semantic segmentation, our goal is to assign a class label to every point in a LiDAR point cloud. Compared to pixel grids of 2D images, point clouds are unordered and sparse, making them more challenging to work with. 3D semantic segmentation has many uses, most notably in autonomous driving. We will discuss PointNet, PointTransformerV3, and Panoptic-Polarnet.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Intro
![2DSeg]({{ '/assets/images/team11/cat_seg.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. Example of 2D Semantic Segmentation* [1].

2D semantic segmentation involves assigning a class label to every pixel in an image. In 3D sematic segmentation, we assign a class label to every point in a LiDAR point cloud. This is a more challenging task due to the sparsity and unordered nature of point clouds, resulting in a lack of inherent spatial relationships between points and making it difficult to utilize traditional grid-based operations like convolutions.

![3DSeg]({{ '/assets/images/team11/3d_seg.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 2. Example of 3D Semantic Segmentation* [2].

Some researchers transform point clouds into voxel grids, a 3D counterpart to pixels. However, voxelization results in an extremely voluminious data representation, making it computationally prohibitive. Instead, we focus on techniques that directly leverage point clouds to perform 3D semantic segmentation.

### LiDAR Point Clouds

### Datasets

### Evalation Metrics

## Models
### PointNet

### PointTransformer V3

### Panoptic-PolarNet


## Reference
[1] Zhou, Bolei. *Lecture 14: Detection + Segmentation.* Computer Science 163: Deep Learning for Computer Vision, 18 Nov. 2024, University of California, Los Angeles. PowerPoint presentation.

[2] Waymo Open Dataset, 17 June 2024. *CVPR Workshop on Autonomous Driving*.

[2] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. "Pointnet: Deep learning on point sets for 3d classification and segmentation." *CVPR*. 2017.

[3] Charles R Qi, Li Yi, Hao Su, and Leonidas J Guibas. "Point-net++: Deep hierarchical feature learning on point sets in a metric space." *NeurIPS*. 2017

[4] Wu, Xiaoyang, Li, Jiang, Peng-Shuai, Wang, Zhĳian, Liu, Xihui, Liu, Yu, Qiao, Wanli, Ouyang, Tong, He, Hengshuang, Zhao. "Point Transformer V3: Simpler, Faster, Stronger." *CVPR*. 2024.

[5] Yang Zhang, Zixiang Zhou, Philip David, Xiangyu Yue, Ze-rong Xi, Boqing Gong, and Hassan Foroosh. "Polarnet: An improved grid representation for online lidar point clouds se-mantic segmentation." *CVPR*. 2020.

[6] Zhou, Zixiang, Yang, Zhang, Hassan, Foroosh. "Panoptic-PolarNet: Proposal-free LiDAR Point Cloud Panoptic Segmentation." *CVPR*. 2021.


---
