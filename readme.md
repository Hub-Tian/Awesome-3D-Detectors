

# List of 3D detection methods

This is a paper and code list of some awesome 3D detection methods. We mainly collect LiDAR-involved methods in autonomous driving. It is worth noticing that we include both official and unofficial codes for each paper. 

![paperlist-map](https://github.com/Hub-Tian/Awesome-3D-Detectors/blob/master/paperlist.png)

## News

*2020.11.18 p.m.* Add **MVAF-Net**

*2020.11.18 p.m.* Add **CADNet** which proposes a context-aware and dynamic feature extraction method to handle the variance of density in point clouds.



## Paper list





| Title                                                        | Pub.               | Input |
| ------------------------------------------------------------ | ------------------ | ----- |
| **MV3D** (Multi-View 3D Object Detection Network for Autonomous Driving) | CVPR2017           | I+L   |
| **F-PointNet** (Frustum PointNets for 3D Object Detection from RGB-D Data) [code](https://github.com/charlesq34/frustum-pointnets) | CVPR2018           | I+L   |
| **VoxelNet** (VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection) | CVPR2018           | L     |
| **PIXOR** (PIXOR: Real-time 3D Object Detection from Point Clouds) [code](https://github.com/philip-huang/PIXOR) | CVPR2018           | L     |
| **AVOD** (Joint 3D Proposal Generation and Object Detection from View Aggregation) [code](https://github.com/kujason/avod) | IROS2018           | I+L   |
| **ContFusion** (Deep Continuous Fusion for Multi-Sensor 3D Object Detection) | ECCV2018           | I+L   |
| **SECOND** (SECOND: Sparsely Embedded Convolutional Detection) [code](https://github.com/traveller59/second.pytorch) | Sensors 2018       | L     |
| **Complex-YOLO** (Complex-YOLO: Real-time 3D Object Detection on Point Clouds) [code](https://github.com/AI-liu/Complex-YOLO) | Axiv2018           | L     |
| **FBF**（Fusing Bird’s Eye View LIDAR Point Cloud and Front View Camera Image for Deep Object Detection）[code](https://github.com/ZiningWang/Sparse_Pooling) | Arxiv2018          | I+L   |
| **RoarNet** (RoarNet: A Robust 3D Object Detection based on Region Approximation Refinement) [code](https://github.com/reinforcementdriving/RoarNet) | IV2019             | I+L   |
| **PVCNN** (Point-Voxel CNN for Efficient 3D Deep Learning) [code](https://github.com/mit-han-lab/pvcnn) | NIPS2019           | L     |
| **MMF**(Multi-Task Multi-Sensor Fusion for 3D Object Detection) [code](https://github.com/facebookresearch/votenet) | CVPR2019           | I+L   |
| **PointPillars** (PointPillars: Fast Encoders for Object Detection from Point Clouds) [code](https://github.com/traveller59/second.pytorch) | CVPR2019           | L     |
| **Point RCNN** (PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud) [code](https://github.com/open-mmlab/OpenPCDet) | CVPR2019           | L     |
| **LaserNet** (LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving) | CVPR2019           | L     |
| **LaserNet++** (Sensor Fusion for Joint 3D Object Detection and Semantic Segmentation) | CVPR2019           | I+L   |
| **Fast PointRCNN **(Fast PointRCNN)                          | ICCV2019           | L     |
| **STD** (STD: Sparse-to-Dense 3D Object Detector for Point Cloud) | ICCV2019           | L     |
| **VoteNet** (Deep Hough Voting for 3D Object Detection in Point Clouds) [code](https://github.com/facebookresearch/votenet) | ICCV2019           | L     |
| **MVX-Net** (MVX-Net: Multimodal VoxelNet for 3D Object Detection) [code](https://github.com/open-mmlab/mmdetection3d) | ICRA2019           | I+L   |
| **Patchs** (Patch Refinement - Localized 3D Object Detection) | Arxiv2019          | L     |
| **StarNet** (StarNet: Targeted Computation for Object Detection in Point Clouds) [code](https://github.com/ModelBunker/StarNet-PyTorch) | Arxiv2019          | L     |
| **F-ConvNet** (Frustum ConvNet: Sliding Frustums to Aggregate Local Point-Wise Features for Amodal 3D Object Detection) | IROS2019           | I+L   |
| **PI-RCNN**（An Efficient Multi-sensor 3D Object Detector with Point-based Attentive Cont-conv Fusion Module） | AAAI2020           | I+L   |
| **TANet** (TANet: Robust 3D Object Detection from Point Clouds with Triple Attention) [code](https://github.com/happinesslz/TANet) | AAAI2020           | L     |
| **MVF** (End-to-end multi-view fusion for 3d object detection in lidar point clouds) [code](https://github.com/AndyYuan96/End-to-End-Multi-View-Fusion-for-3D-Object-Detection-in-LiDAR-Point-Clouds) | ICRL2020           | L     |
| **SegVoxelNet** (SegVoxelNet: Exploring Semantic Context and Depth-aware Features for 3D Vehicle Detection from Point Cloud) | ICRA2020           | L     |
| **Voxel-FPN** (Voxel-FPN: multi-scale voxel feature aggregation in 3D object detection from point clouds) | Sensors 2020       | L     |
| **AA3D** (Adaptive and Azimuth-Aware Fusion Network of Multimodal Local Features for 3D Object Detection) | Neurocomputing2020 | I+L   |
| **Part A^2** (Part-A^ 2 Net: 3D Part-Aware and Aggregation Neural Network for Object Detection from Point Cloud) [code](https://github.com/open-mmlab/OpenPCDet) | TPAMI2020          | L     |
| **PV-RCNN** (PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection) [code](https://github.com/open-mmlab/OpenPCDet) | CVPR2020           | L     |
| **3D SSD** (3DSSD: Point-based 3D Single Stage Object Detector) [code](https://github.com/Jia-Research-Lab/3DSSD) | CVPR2020           | L     |
| **Associate-3Ddet** (Associate-3Ddet: Perceptual-to-Conceptual Association for 3D Point Cloud Object Detection) [code](https://github.com/dleam/Associate-3Ddet) | CVPR2020           | L     |
| **HVNet** (HVNet: Hybrid Voxel Network for LiDAR Based 3D Object Detection) [code](https://github.com/AndyYuan96/HVNet) | CVPR2020           | L     |
| **ImVoteNet** (ImVoteNet: Boosting 3D Object Detection in Point Clouds with Image Votes) | CVPR2020           | I+L   |
| **Point GNN** (Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud) | CVPR2020           | L     |
| **SA-SSD** (Structure Aware Single-stage 3D Object Detection from Point Cloud) [code](https://github.com/skyhehe123/SA-SSD) | CVPR2020           | L     |
| (What You See is What You Get: Exploiting Visibility for 3D Object Detection) | CVPR2020           | L     |
| **DOPS** (DOPS: Learning to Detect 3D Objects and Predict their 3D Shapes) | CVPR2020           | L     |
| **3D IoU-Net** (3D IoU-Net: IoU Guided 3D Object Detector for Point Clouds) | Arxiv2020          | L     |
| **3D CVF** (3D-CVF: Generating Joint Camera and LiDAR Features Using Cross-View Spatial Feature Fusion for 3D Object Detection) | ECCV2020           | I+L   |
| **HotSpotNet** (Object as Hotspots: An Anchor-Free 3D Object Detection Approach via Firing of Hotspots) | ECCV2020           | L     |
| **EPNet**: (EPNet: Enhancing Point Features with Image Semantics for 3D Object Detection) [code](https://github.com/happinesslz/EPNet) | ECCV2020           | I+L   |
| **WS3D** (Weakly Supervised 3D Object Detection from Lidar Point Cloud) [code](https://github.com/hlesmqh/WS3D) | ECCV2020           | L     |
| **Pillar-OD** Pillar-based Object Detection for Autonomous Driving [code](https://github.com/WangYueFt/pillar-od) | ECCV2020           | L     |
| **SSN** (SSN: Shape Signature Networks for Multi-class Object Detection from Point Clouds) | Arxiv2020          | L     |
| **CenterPoint** (Center-based 3D Object Detection and Tracking) [code](https://github.com/tianweiy/CenterPoint) | Arxiv2020          | L     |
| **AFDet** (AFDet: Anchor Free One Stage 3D Object Detection) | Waymo2020          | L     |
| **LGR-Net** (Local Grid Rendering Networks for 3D Object Detection in Point Clouds) | arxiv2020.07       | L     |
| **CenterNet3D** (CenterNet3D:An Anchor free Object Detector for Autonomous Driving)[code](https://github.com/wangguojun2018/CenterNet3d) | arxiv2020.07       | L     |
| **RCD** (Range Conditioned Dilated Convolutions for Scale Invariant 3D Object Detection) | arxiv2020.06       | L     |
| **VS3D** (Weakly Supervised 3D Object Detection from Point Clouds) [code](https://github.com/Zengyi-Qin/Weakly-Supervised-3D-Object-Detection) | ACM MM2020         | I+L   |
| **LC-MV** (Multi-View Fusion of Sensor Data for Improved Perception and Prediction in Autonomous Driving) | CoRL2020           | I+L   |
| **RangeRCNN** (RangeRCNN: Towards Fast and Accurate 3D Object Detection with Range Image Representation) | arxiv2020.09       | L     |
| **MVAF-Net** (Multi-View Adaptive Fusion Network for 3D Object Detection) | arxiv2020.11       | I+L   |
| **CADNet** ([Context-Aware Dynamic Feature Extraction for 3D Object Detection in  Point Clouds](http://arxiv.org/abs/1912.04775v3)) | arxiv2020.07       | L     |
| To be continued...                                           |                    |       |

## Code list

- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) in **pytorch**

  Methods supported: **SECOND, PointPillars, FreeAnchor, VoteNet, Part-A2, MVXNet**

  Benchmark supported: **KITTI**, **nuScenes**, **Lyft**, **ScanNet**, **SUNRGBD**

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet): An open source project for LiDAR-based 3D scene perception in **Pytorch**.

  Methods supported : **PointPillars**, **SECOND**, **Part A^2**, **PV-RCNN**, PointRCNN(ongoing).

  Benchmark supported: **KITTI**, Waymo (ongoing).

- [Det3d](https://github.com/poodarchu/det3d): A general 3D Object Detection codebase in **PyTorch**. 

  Methods supported : **PointPillars**, **SECOND**, **PIXOR**.

  Benchmark supported: **KITTI**, **nuScenes**, **Lyft**.

- [second.pytorch](https://github.com/traveller59/second.pytorch): SECOND detector in **Pytorch**.

  Methods supported : **PointPillars**, **SECOND**.

  Benchmark supported: **KITTI**, **nuScenes**.

- [CenterPoint](https://github.com/tianweiy/CenterPoint): "*Center-based 3D Object Detection and Tracking*" in **Pytorch**.

  Methods supported : **CenterPoint-Pillar**, **Center-Voxel**.

  Benchmark supported:  **nuScenes**，**Waymo**.

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

- [Complex-YOLOv4-pytorch](https://github.com/maudzung/Complex-YOLOv4-Pytorch): " Complex-YOLO: Real-time 3D Object Detection on Point Clouds)" in **pytorch**.

  Methods supported : **YOLO**

  Benchmark supported: **KITTI**.

- [EPNet](https://github.com/happinesslz/EPNet): "EPNet: Enhancing Point Features with Image Semantics for 3D Object Detection  "

  Methods supported: **EPNet**

  Benchmark supported: **KITTI**, **SUN-RGBD**

- [Super Fast and Accurate 3D Detector](https://github.com/maudzung/Super-Fast-Accurate-3D-Object-Detection):"Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds"

  Benchmark supported: **KITTI**


## Dataset list

(reference: https://mp.weixin.qq.com/s/3mpbulAgiwi5J66MzPNpJA from WeChat official account: "CNNer")

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
