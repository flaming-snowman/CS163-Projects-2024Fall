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

Semantic segmentation is the backbone of many understanding tasks, from robotics to medical autonomy. 2D semantic segmentation involves assigning a class label to every pixel in an image. In 3D sematic segmentation, we extend this to assigning a class label to every point in a LiDAR point cloud. This is a much more challenging task due to the sparsity and unordered nature of point clouds, removing the helpful spatial locality and inherent density of 2D images we relied on to utilize traditional grid-based operations efficiency. Because of a lack of inherent spatial relationship between an unordered point set, we can't naively process them through techniques like convolutions or patching. We discuss 3 deep learning based approaches to this challeng. 

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

PointNet was one of the first deep learning solutions to directly process raw point clouds without requiring a voxelization or projection step into a structure input. It addressed three key challenges: 
- Permutation invariance: point cloud data is inherently unordered and should handle all $$N!$$ possible permutations of the input exactly the same.
- Interaction among points: in acknolwedging permutation invariance and foregoing a direct 3-dimensional structural representation, we should still account for the distance between points to maintain spatial information. For example, we might imagine as in CNNs that local features aggregate and build up to more complex, semantic features.
- Transformation invariance: a geometric object's representations and downstream predictions should ideally be identical under transformations like rotations.
In addressing these, PointNet became the new state-of-the-art as a backbone for 3D recognitions tasks, ranging from classification to part and instance segmentation, showing better robustness to rigid motions and perturbations. 

The key to PointNets lies in finding an informative symmetic function. A symmetric function $$g$$ is one that approximates a set on a function $$f$$ as:

We approximate the function applied to a set by applying a symmetric function on transformed elements in the set:
$$
f(\{x_1, \dots, x_n\}) \approx g(h(x_1), \dots, h(x_n)),
$$

where $$f : 2^{\mathbb{R}^N} \to \mathbb{R}$$ is the module we develop, $$h : \mathbb{R}^N \to \mathbb{R}^K $$ is a transformation we're learning (to extract features from the data), and  
$$ \gamma  : \underbrace{\mathbb{R}^K \times \cdots \times \mathbb{R}^K}_{n} \to \mathbb{R} $$ is a symmetric function we rap these transformed outputs in to get a permutation invariant output. Instead, we can simply use an elementwise maximum function:
$$
f(\{x_1, \ldots, x_n\}) \approx \gamma (\operatorname{MAX}_{i=1,\ldots,n}\{h(x_i)\})
$$
This is both very computationally efficient and empirically powerful. Intuitively, it tries captures the most important signals. $$gamma$$ and $$h$$ are typically MLP networks to enable universal set function approximation, as we describe below.

PointNet also captures both global and local features for segmentation. It learns both a $$1024$$-dimensional global feature vector $$F$$ and $$64$$-dimensional pointwise feature vectors $$f_i$$, which are concatenated together before being passed into our segmentation head. This can improve performance on dense prediction tasks that rely on local features for final predictions.

Lastly, to address transformation invariance, one simply solution would be to just align an input set to a known orientation before feature extraction. This can be learned as an affine transformation, learned by a "T-net", a smaller submdoule that's a neural network on its own. This network's architecture is similar to the overall network ifself, learning an affine transformation matrix $$A$$ for canonicalizing input point clouds. We also use $$\ell_2$$ regularization to keep the matrix orthogonal, which can prevent compressing space and losing information. 
$$
\ell=\lVert I-AA^T\rVert_F^2
$$
This empirically stabilized the training trajectory and convergence speed.

The authors of the PointNet paper also showed that PointNet maintained the universal approximation property--it can approximate any continuous set function on point clouds. Intuitively, we can interpret the model to show that it learns by summarizing shapes with a skeleton, a sparse subset of informative points. This guides the model to select interesting and informative points, regardless of their exact positions in 3D space, ane encode its reason for selecting them as a backbone. (TODO SHORTEN PARAGRAPH) This improves robustness by only relying on a finite set of critical points per object, since as long as these critical points are unchanged the model is robust to other noise and occlusion. 




RNNs - O. Vinyals, S. Bengio, and M. Kudlur. Order matters: Sequence to sequence for sets. arXiv preprint
arXiv:1511.06391, 2015.


PointNet scales linearly at $$\mathcal{O}(n)$$ compute with respect to the number of points, while 3D convolutions would scale at $$\mathcal{O}(n^3)$$!

### PointNet++

While PointNet revolutioned deep learning in 3D at scale, it is not without its limitations. Particularly, its naively relies solely on MLP layers, lacking the locality and hierarchical feature extraction that made CNNs effective in both 2D and 3D cases, even if inefficient. PointNet is also unable to handle non-uniform LiDAR scans, assuming that its points are uniformly distributed. However, this is rarely true as real world point cloud have varying densities. For example, an autonomous vehicle will have greatly increased densities by nearby objects like pedestrians and street signs, but a low point density on distance roads, posing a challenge for PointNet to understand the environment's structure. 

PointNet++ addresses these challenges by introducing hierarchical learning in three layers:
1. First, a sampling layer uses farthest point sampling to select representative centroids for local regions. FPS chooses a subset $$\{x_1{i_1},x_{i_2},\ldots,x_{i_m}\}$$ such that $$x_{i_j}$$ is the most distant point from the the points $$\{x_1{i_1},x_{i_2},\ldots,x_{i_{j-1}}\}$$ with respect to the rest of the points. 
2. Second, a grouping layer construct $$N'\times K\times (d+C)$$ local regions around the $$N'$$ $$d$$-dimensional centroids from an $$N\times (d+C)$$ point set, where $$K$$ is the number of neighbors to group together. This is done either via k-nearest neighbors sellecion, or a ball query where all points within a radius to a query point are added to a subgroup. 
3. A local mini-PointNet is applied to each region to extract features. Local geometry is encoded via relative coordinates.

This transforms the problem into smaller, more uniformly distributed sections that PointNet can handle, to be much more robust under non-uniform samling density. Similar to the inductive bias of CNNs, PointNet++ learns features at multiple set abstraction levels, where the above process is repeated many times to produce sets of fewer elements. This enables the model to learn fine-grained features and capture local geometry, but also provide contextual information via pooling layers. 

Additionaly, to handle the non-uniform sampling sampling inherent to point cloud data, PointNet++ introduces two new density adaptive layers. The first, Multi-Scale Grouping, uses a random dropout with a random rate of $$\theta\sim U(0, p)$$ for $$o\leq 1$$ on inputs to enforce training on data of various sparsities and various uniformities. This essentially extends on traditional neural network dropout by learning levels of sparsity, maintaining the same benefits like partially disentangling neurons. As this method is computationally expensive, since it would require running an entire PointNet forward pass even for very small centroids regions, PointNet++ also introduces Multi-resolution grouping, which essentially joins features from low density regions in a weighted manner.

Finally, they propose point feature propagation, where point features are propagated back into the original point cloud via an inverse distance-weighted interpolation based on $k$ nearest neighbors, which are connected from the set abstraction back into point space with skip connections. Similar to ResNet residual connections, this helps the model make high-resolution predictions. 

PointNet++ set a new state-of-the-art for semantic segmentation, object detection, and classification tasks in 3D, ourperforming PointNet on small datasets like ScanNet and ModelNet40 of the time. By exploiting inherent hierarchies in data more efficiently than techniques like 3D convolutions or even sparse convolutions, PointNet++ addressed many of the original assumptions PointNet made about its data, improving robustness. However, its computational costs are stil much higher. 

## PointTransformer V3

PointTransformer V3 is the current state-of-the-art for 3D segmentation tasks, holding the highest accuracy on benchmarks like SemanticKITTI ScanNet as of today.

PointTransformer V1 marked 3D's "ViT moment," tailoring the attention mechanism for point clouds. Unlike PointNet's reliance on MLPs, PointTransformer uses attention to share rich local and global features through unstructed point clouds. However, its reliance on $$k$$ nearest neighbors and pairwise computation as a whole rendered it difficult to scale. 

Insert PointTransformer V1 picture

PointTransformer V2 built on top of this by introducing Grouped Vector Attention (GVA) to enahnce computational efficiency. It presented a series of small optimizations on top of the original PointTransformer that added up, such as halving the total pairwise computations, increasing overall performance but reamining limited for large-scale dataset

Insert PointTransformer V2 picture

PointTransformer V3 (PTv3) simplifies the architecture further and focuses on scale. One main focus of the project is point cloud serialization, a much more efficient method to transform unstructured point cloud data into a structured, serialized format for computation. This is done via space-filling curves, paths through a discrete high-dimensional space (like point space). 

PTv3 focuses specifically on Z-order curves and Hilbert curves, both of which preserve spatial locality and improve computational efficiency by structuring data. They trasvers traverse 3D space through their specificied path, capturing discovered point clouds into a serialized 1D sequence. To leverage this locality preservations, the serlaization process assigns points basd on their position into a serialization code. This is similar to the embedding process for traditional transformers. 

fedwadwadwa

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
