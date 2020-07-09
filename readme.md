

# List of 3D detection methods

This is a paper and code list of some awesome 3D detection methods. We mainly collect LiDAR-involved methods in autonomous driving. 

## Paper list

| Title                                                        | Pub.               | Input |
| ------------------------------------------------------------ | ------------------ | ----- |
| **MV3D** (Multi-View 3D Object Detection Network for Autonomous Driving) | CVPR2017           | I+L   |
| **F-PointNet** (Frustum PointNets for 3D Object Detection from RGB-D Data) | CVPR2018           | I+L   |
| **VoxelNet** (VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection) | CVPR2018           | L     |
| **PIXOR** (PIXOR: Real-time 3D Object Detection from Point Clouds) | CVPR2018           | L     |
| **AVOD** (Joint 3D Proposal Generation and Object Detection from View Aggregation) | IROS2018           | I+L   |
| **ContFusion** (Deep Continuous Fusion for Multi-Sensor 3D Object Detection) | ECCV2018           | I+L   |
| **SECOND** (SECOND: Sparsely Embedded Convolutional Detection) | Sensors 2018       | L     |
| **RoarNet** (RoarNet: A Robust 3D Object Detection based on Region Approximation Refinement) | IV2019             | I+L   |
| **PVCNN** (Point-Voxel CNN for Efficient 3D Deep Learning)   | NIPS2019           | L     |
| **MMF** (Multi-Task Multi-Sensor Fusion for 3D Object Detection) | CVPR2019           | I+L   |
| **PointPillars** (PointPillars: Fast Encoders for Object Detection from Point Clouds) | CVPR2019           | L     |
| **Point RCNN** (PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud) | CVPR2019           | L     |
| **LaserNet** (LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving) | CVPR2019           | L     |
| **LaserNet++** (Sensor Fusion for Joint 3D Object Detection and Semantic Segmentation) | CVPR2019           | I+L   |
| **Fast PointRCNN **(Fast PointRCNN)                          | ICCV2019           | L     |
| **STD** (STD: Sparse-to-Dense 3D Object Detector for Point Cloud) | ICCV2019           | L     |
| **VoteNet** (Deep Hough Voting for 3D Object Detection in Point Clouds) | ICCV2019           | L     |
| **MVX**-Net (MVX-Net: Multimodal VoxelNet for 3D Object Detection) | ICRA2019           | I+L   |
| **Patchs** (Patch Refinement - Localized 3D Object Detection) | Arxiv2019          | L     |
| **StarNet** (StarNet: Targeted Computation for Object Detection in Point Clouds) | Arxiv2019          | L     |
| **F-ConvNet** (Frustum ConvNet: Sliding Frustums to Aggregate Local Point-Wise Features for Amodal 3D Object Detection) | IROS2019           | I+L   |
| **PI-RCNN**: An Efficient Multi-sensor 3D Object Detector with Point-based Attentive Cont-conv Fusion Module | AAAI2020           | I+L   |
| **TANet** (TANet: Robust 3D Object Detection from Point Clouds with Triple Attention) | AAAI2020           | L     |
| **MVF** (End-to-end multi-view fusion for 3d object detection in lidar point clouds) | ICRL2020           | L     |
| **SegVoxelNet** (SegVoxelNet: Exploring Semantic Context and Depth-aware Features for 3D Vehicle Detection from Point Cloud) | ICRA2020           | L     |
| **Voxel-FPN** (Voxel-FPN: multi-scale voxel feature aggregation in 3D object detection from point clouds) | Sensors 2020       | L     |
| **AA3D** (Adaptive and Azimuth-Aware Fusion Network of Multimodal Local Features for 3D Object Detection) | Neurocomputing2020 | I+L   |
| **Part A^2** (Part-A^ 2 Net: 3D Part-Aware and Aggregation Neural Network for Object Detection from Point Cloud) | TPAMI2020          | L     |
| **PV-RCNN** (PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection) | CVPR2020           | L     |
| **3D SSD** (3DSSD: Point-based 3D Single Stage Object Detector) | CVPR2020           | L     |
| **Associate-3Ddet** (Associate-3Ddet: Perceptual-to-Conceptual Association for 3D Point Cloud Object Detection) | CVPR2020           | L     |
| **HVNet** (HVNet: Hybrid Voxel Network for LiDAR Based 3D Object Detection) | CVPR2020           | L     |
| **ImVoteNet** (ImVoteNet: Boosting 3D Object Detection in Point Clouds with Image Votes) | CVPR2020           | I+L   |
| **Point GNN** (Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud) | CVPR2020           | L     |
| **SA-SSD** (Structure Aware Single-stage 3D Object Detection from Point Cloud) | CVPR2020           | L     |
| (What You See is What You Get: Exploiting Visibility for 3D Object Detection) | CVPR2020           | L     |
| **3D IoU-Net** (3D IoU-Net: IoU Guided 3D Object Detector for Point Clouds) | Arxiv2020          | L     |
| **3D CVF** (3D-CVF: Generating Joint Camera and LiDAR Features Using Cross-View Spatial Feature Fusion for 3D Object Detection) | ECCV2020           | I+L   |
| **HotSpotNet** (Object as Hotspots: An Anchor-Free 3D Object Detection Approach via Firing of Hotspots) | ECCV2020           | L     |
| **SSN** (SSN: Shape Signature Networks for Multi-class Object Detection from Point Clouds) | Arxiv2020          | L     |
| **CenterPoint** (Center-based 3D Object Detection and Tracking) | Arxiv2020          | L     |
| **AFDet** (AFDet: Anchor Free One Stage 3D Object Detection) | Waymo2020          | L     |
| To be continued...                                           |                    |       |

## Code list

- [Det3d](https://github.com/poodarchu/det3d): A general 3D Object Detection codebase in **PyTorch**. 

  Methods supported : **PointPillars**, **SECOND**, **PIXOR**.

  Benchmark supported: **KITTI**, **nuScenes**, **Lyft**.

- [second.pytorch](https://github.com/traveller59/second.pytorch): SECOND detector in **Pytorch**.

  Methods supported : **PointPillars**, **SECOND**.

  Benchmark supported: **KITTI**, **nuScenes**.
  
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet): An open source project for LiDAR-based 3D scene perception in **Pytorch**.

  Methods supported : **PointPillars**, **SECOND**, **Part A^2**, **PV-RCNN**, PointRCNN(ongoing).

  Benchmark supported: **KITTI**, Waymo (ongoing).
  
- [CenterPoint](https://github.com/tianweiy/CenterPoint): "*Center-based 3D Object Detection and Tracking*" in **Pytorch**.

  Methods supported : **CenterPoint-Pillar**, **Center-Voxel**.

  Benchmark supported:  **nuScenes**.

- [SA-SSD](https://github.com/skyhehe123/SA-SSD): "SA-SSD: *Structure Aware Single-stage 3D Object Detection from Point Cloud*" in **pytorch**

  Methods supported : **SA-SSD**.
  
  Benchmark supported: **KITTI**.
  
- [3DSSD](https://github.com/Jia-Research-Lab/3DSSD): "*Point-based 3D Single Stage Object Detector* " in **Tensorflow**.

  Methods supported : **3DSSD**, **PointRCNN**, STD (ongoing).

  Benchmark supported: **KITTI**, nuScenes (ongoing).

- [Point-GNN](https://github.com/WeijingShi/Point-GNN): "Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud" in **Tensorflow**.

  Methods supported : **Point-GNN**.

  Benchmark supported: **KITTI**.

- [TANet](https://github.com/happinesslz/TANet): "*TANet: Robust 3D Object Detection from Point Clouds with Triple Attention*" in **Pytorch**.

  Methods supported : **TANet** (PointPillars, Second).

  Benchmark supported: **KITTI**.

## Dataset list

(reference: https://mp.weixin.qq.com/s/3mpbulAgiwi5J66MzPNpJA from WeChat official account: "CNNer"*)

- **KITTI**

  Website: http://www.cvlibs.net/datasets/kitti/raw_data.php

  Paper: http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

- **Waymo**

  Website: https://waymo.com/open

  Paper: https://arxiv.org/abs/1912.04838v5

- **NuScenes**

  Website: https://www.nuscenes.org/

  Paper: https://arxiv.org/abs/1903.11027

- **Lyft**

  Website: https://level5.lyft.com/

  Paper: https://level5.lyft.com/dataset/

- **Audi autonomous driving dataset**

  Website: http://www.a2d2.audi

  Paper: https://arxiv.org/abs/2004.06320

- **Apollo**

  Website: http://apolloscape.auto/

  Paper: https://arxiv.org/pdf/1803.06184.pdf

  

  