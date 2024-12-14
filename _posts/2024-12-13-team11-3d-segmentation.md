---
layout: post
comments: true
title: 3D Semantic Segmentation
author: Rathul Anand, Jason Liu
date: 2024-12-12
---

> 3D semantic segmentation is a cornerstone of modern computer vision, enabling understanding of our physical world for applications ranging from embodied intelligence and robotics to autonomous driving. In 3D semantic segmentation, our goal is to assign a semantic label to every point in a LiDAR point cloud. Compared to pixel grids of 2D images, data from 3D sensors are complex, irregular, and sparse, lacking the niceties and biases in data we often exploit in processing 2D images. We will discuss 3 deep learning based approaches pioneering the field, namely PointNet, PointTransformerV3, and Panoptic-Polarnet.

<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

# Introduction
![2DSeg]({{ '/assets/images/team11/cat_seg.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. Example of 2D Semantic Segmentation* [1].

Semantic segmentation is a foundational task in computer vision for scene understanding. 2D semantic segmentation involves assigning a class label to every pixel in an image. In 3D sematic segmentation, we assign a class label to every point in a LiDAR point cloud. This is a more challenging task due to the sparsity and unordered nature of point clouds, resulting in a lack of inherent spatial relationships between points and making it difficult to utilize traditional grid-based operations. While we often relied on feautres like spatial locality and the inherent density of 2D images to process them through techniques like convolutions or patches, we must tackle the irregularity and sparsity of most 3D representations. 

![3DSeg]({{ '/assets/images/team11/3d_seg.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 2. Example of 3D Semantic Segmentation* [2].

Some researchers transform point clouds into voxel grids, a 3D counterpart to pixels. However, voxelization results in an extremely voluminious data representation, making it computationally prohibitive. Instead, we focus on techniques that directly leverage point clouds to perform 3D semantic segmentation.

## 3D Scene Representations
### Point Clouds
We often represent 3D scenes as point clouds. A point cloud is a collection of $$n$$ unordered points in 3D space: 

$$
    \mathcal{P}=\{(x_i, y_i, z_i, f_i) : i=1,\ldots,n\}
$$

where each point is associated with a spatial location and $$k$$-dimensional $$f_i$$ vector encoding optional features like color and intensity, and eventually semantic label. Typically, we represent this as a matrix $$\mathbf{P}\in\mathbb{R}^{n\times (3+k)}$$. 

Insert point cloud picture here

Point clouds are often collected from LiDAR sensors, which use the time of flight to obtain the 3D shape of a surrounding environment. LiDAR sensors shoot out impulses (via a packet of photons—-rays) to a scene, each of which bounce off an object $$z$$ meters away. Given the time to return $$t_d$$ and the speed of light, we can calculate

$$
    z=c\times t_d\times 0.5
$$

to get the distance each ray traveled before bouncing back. Combining with the ray angle data, we can capture a point cloud of the surface observed.

Insert LiDAR picture here

### Voxel Grids
Voxels are the most natural extension of pixels into the 3-dimensional case, as they're essentially 3D pixels. A voxel is a volumetric representation dividing a space into uniform cubes, represented as a 4-dimensional tensor containing information about each $$(x,y,z)$$ location in a scene:

$$
    \mathbf{V}\in\mathbb{R}^{X\times Y\times Z\times C}
$$

where each channel might contain RGB data, reflectance, and instance-level data like semantic labels. 

Voxel size determines the grid's resolution, which can lead to a tradeoff between high resolution and the number of unused voxels. Given that we typically only care about a 2D manifold/surface within the 3D space, voxels will capture mostly empty cubes, which can naturally lead to challenges with naively applying traditional operations like convolution. For this reason, we often work with binary voxels, tensors containing $$1$$'s for occupied space and $$0$$'s for empty space. Despite the benefit of spatial ordering that voxels give us, for this reason voxels will suffer from a significant computational overhead to process.

### Meshes
Meshes represent 3D objects as a collection of vertices, edges, and faces. This better aligns with the features we wish to learn from in 3D space, as they contain rich geometric and topological information. However, they're irregular and more difficult to represent in a neural network, leading to extra complexity in processing and reconstruction.

### Depth Images and BEV
Depth images and approaches like Bird's Eye View fusion are approaches that project 3D data into 2D formats for computational simplicity. They often work alongside LiDAR-caera fusion for planning tasks. TODO

## Datasets and Evaluation Metrics
### Datasets TODO add history
The Stanford Large-Scale 3D Indoor Spaces (S3DIS) datset contains 3D point clouds of indoor environments like offices and conference rooms. The dataset contains 6 large-scale with 271 rooms. There are 13 semantic classes to learn from.

Insert image

nuScenes is a large-scale autonomous driving dataset, containing 1000 scenes that each last 20 seconds. nuScenes is collected using a full sensor suite of LiDAR, radar, gamera, GPS, and IMU, lending itself well to a variety of self-driving tasks and settings. Semantic and instance segmentation labels are provided.

Insert image

The Waymo Open Dataset is another large-scale autonomous driving dataset, containing LiDAR and camera data with 3D semantic labels. SemanticKITTI is another dataset that focuses on semantic and instance segmentation for urban street scenes.

Insert image
### Evaluation Metrics
As we did in 2D segmentation, we're parimarily concerned with the Intersection over Union (IoU) score of our models. This measures the overal between the predicted segmentation and the ground truth segmentation for each class. For a scene and class, we can calculate:

$$
    \text{IoU}=\frac{|\text{Predicted}\cap\text{Ground Truth}}{\text{Predicted}\cup\text{Ground Truth}}
$$

Our goal is to minimize mIoU, the mean IoU over all classes.

For panoptic segmentation tasks, we also measure Panoptic Quality (PO). This measure combines segmentation quality (IoU) with recognition quantity (often F1 score, measuring precision and recall):

$$
    \text{PQ}=\text{SQ}\cdot \text{RQ}
$$

## Motivation and Applications
TODO
Scene understanding
Navigation and path planning
Obstacle avoidance
Object detection, tracking, motion planning, classification
Robotics
Urban mapping

## Traditional Approaches
Early methods often drew heavily from 2D segmentation solutions, converting point clouds into voxel grids and applying 3D convolutional neural networks. However, voxelization has a drawback of high memory consumption and computational inefficiency, even when sparse methods were introduced, in comparison to point cloud-based methods (TODO: SOURCE?). Multi-view approaches relying on systems like BEVFusion also focused on projecting 3D data into multiple 2D views for easier processing by relying on our existing 2D segmentation systems, but this also comes at a loss of spatial information due to projection. (CITATION)

For this reason, we'll explore solutions that operate directly on point clouds, which are more spatially efficient. 

# Deep Learning Approaches

## PointNet

PointNet was one of the first deep learning solutions to directly process raw point clouds without requiring a voxelization or projection step into a structure input. It addressed the key challenge of permutation invariance, developing a technique that enabled a model to learn outputs that are the sme regardless of the order points are inputted. 

### PointNet++


## PointTransformerV3

## Panoptic-PolarNet

## Results
# Analysis
## Comparisons
## Impact and Future Directions
## Conclusions

# References
[1] Zhou, Bolei. *Lecture 14: Detection + Segmentation.* Computer Science 163: Deep Learning for Computer Vision, 18 Nov. 2024, University of California, Los Angeles. PowerPoint presentation.

[2] Waymo Open Dataset, 17 June 2024. *CVPR Workshop on Autonomous Driving*.

[2] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. "Pointnet: Deep learning on point sets for 3d classification and segmentation." *CVPR*. 2017.

[3] Charles R Qi, Li Yi, Hao Su, and Leonidas J Guibas. "Point-net++: Deep hierarchical feature learning on point sets in a metric space." *NeurIPS*. 2017

[4] Wu, Xiaoyang, Li, Jiang, Peng-Shuai, Wang, Zhĳian, Liu, Xihui, Liu, Yu, Qiao, Wanli, Ouyang, Tong, He, Hengshuang, Zhao. "Point Transformer V3: Simpler, Faster, Stronger." *CVPR*. 2024.

[5] Yang Zhang, Zixiang Zhou, Philip David, Xiangyu Yue, Ze-rong Xi, Boqing Gong, and Hassan Foroosh. "Polarnet: An improved grid representation for online lidar point clouds se-mantic segmentation." *CVPR*. 2020.

[6] Zhou, Zixiang, Yang, Zhang, Hassan, Foroosh. "Panoptic-PolarNet: Proposal-free LiDAR Point Cloud Panoptic Segmentation." *CVPR*. 2021.


---
