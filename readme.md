# List of 3D detection methods

This is a paper and code list of some awesome 3D detection methods. We mainly collect LiDAR-involved methods in autonomous driving. It is worth noticing that we include both official and unofficial codes for each paper. 

![paperlist-map](https://github.com/Hub-Tian/Awesome-3D-Detectors/blob/master/paperlist.png)

## News

*2022.2.17* Add the link for every paper. (Happy Chinese New Year!)

## Paper list

| Title                                                        | code                                                         | Pub.               | Input |
| :----------------------------------------------------------- | ------------------------------------------------------------ | ------------------ | ----- |
| **MV3D** ([Multi-View 3D Object Detection Network for Autonomous Driving](https://ieeexplore.ieee.org/document/8100174)) |                                                              | CVPR2017           | I+L   |
| **F-PointNet** ([Frustum PointNets for 3D Object Detection from RGB-D Data](https://ieeexplore.ieee.org/document/8578200)) | [code](https://github.com/charlesq34/frustum-pointnets)      | CVPR2018           | I+L   |
| **VoxelNet** ([VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://ieeexplore.ieee.org/document/8578570)) |                                                              | CVPR2018           | L     |
| **PIXOR** ([PIXOR: Real-time 3D Object Detection from Point Clouds](https://ieeexplore.ieee.org/document/8578896)) | [code](https://github.com/philip-huang/PIXOR)                | CVPR2018           | L     |
| **AVOD** ([Joint 3D Proposal Generation and Object Detection from View Aggregation](https://ieeexplore.ieee.org/abstract/document/8594049)) | [code](https://github.com/kujason/avod)                      | IROS2018           | I+L   |
| **ContFusion** ([Deep Continuous Fusion for Multi-Sensor 3D Object Detection](https://openaccess.thecvf.com/content_ECCV_2018/html/Ming_Liang_Deep_Continuous_Fusion_ECCV_2018_paper.html)) |                                                              | ECCV2018           | I+L   |
| **SECOND** ([SECOND: Sparsely Embedded Convolutional Detection](https://pubmed.ncbi.nlm.nih.gov/30301196/n)) | [code](https://github.com/traveller59/second.pytorch)        | Sensors 2018       | L     |
| **Complex-YOLO** ([Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://arxiv.org/abs/1803.06199)) | [code](https://github.com/AI-liu/Complex-YOLO)               | Axiv2018           | L     |
| **FBF**（[Fusing Bird’s Eye View LIDAR Point Cloud and Front View Camera Image for Deep Object Detection](https://ieeexplore.ieee.org/abstract/document/8500387)） | [code](https://github.com/ZiningWang/Sparse_Pooling)         | IV2018             | I+L   |
| **RoarNet** ([RoarNet: A Robust 3D Object Detection based on Region Approximation Refinement](https://ieeexplore.ieee.org/abstract/document/8813895)) | [code](https://github.com/reinforcementdriving/RoarNet)      | IV2019             | I+L   |
| **PVCNN** ([Point-Voxel CNN for Efficient 3D Deep Learning](https://arxiv.org/abs/1907.03739)) | [code](https://github.com/mit-han-lab/pvcnn)                 | NIPS2019           | L     |
| **MMF**([Multi-Task Multi-Sensor Fusion for 3D Object Detection](https://openaccess.thecvf.com/content_CVPR_2019/html/Liang_Multi-Task_Multi-Sensor_Fusion_for_3D_Object_Detection_CVPR_2019_paper.html)) | [code](https://github.com/facebookresearch/votenet)          | CVPR2019           | I+L   |
| **PointPillars** ([PointPillars: Fast Encoders for Object Detection from Point Clouds](https://ieeexplore.ieee.org/document/8954311)) | [code](https://github.com/traveller59/second.pytorch)        | CVPR2019           | L     |
| **Point RCNN** ([PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud](https://arxiv.org/abs/1812.04244)) | [code](https://github.com/open-mmlab/OpenPCDet)              | CVPR2019           | L     |
| **LaserNet** ([LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving](https://openaccess.thecvf.com/content_CVPR_2019/papers/Meyer_LaserNet_An_Efficient_Probabilistic_3D_Object_Detector_for_Autonomous_Driving_CVPR_2019_paper.pdf)) |                                                              | CVPR2019           | L     |
| **LaserNet++** ([Sensor Fusion for Joint 3D Object Detection and Semantic Segmentation](https://openaccess.thecvf.com/content_CVPRW_2019/html/WAD/Meyer_Sensor_Fusion_for_Joint_3D_Object_Detection_and_Semantic_Segmentation_CVPRW_2019_paper.html)) |                                                              | CVPR2019           | I+L   |
| **Fast PointRCNN **([Fast PointRCNN](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Fast_Point_R-CNN_ICCV_2019_paper.html)) |                                                              | ICCV2019           | L     |
| **STD** ([STD: Sparse-to-Dense 3D Object Detector for Point Cloud](https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_STD_Sparse-to-Dense_3D_Object_Detector_for_Point_Cloud_ICCV_2019_paper.html)) |                                                              | ICCV2019           | L     |
| **VoteNet** ([Deep Hough Voting for 3D Object Detection in Point Clouds](https://ieeexplore.ieee.org/document/9008567)) | [code](https://github.com/facebookresearch/votenet)          | ICCV2019           | L     |
| **MVX-Net** ([MVX-Net: Multimodal VoxelNet for 3D Object Detection](https://ieeexplore.ieee.org/abstract/document/8794195)) | [code](https://github.com/open-mmlab/mmdetection3d)          | ICRA2019           | I+L   |
| **Patchs** ([Patch Refinement - Localized 3D Object Detection](https://arxiv.org/abs/1910.04093)) |                                                              | Arxiv2019          | L     |
| **StarNet** ([StarNet: Targeted Computation for Object Detection in Point Clouds](https://arxiv.org/abs/1908.11069)) | [code](https://github.com/ModelBunker/StarNet-PyTorch)       | Arxiv2019          | L     |
| **F-ConvNet** ([Frustum ConvNet: Sliding Frustums to Aggregate Local Point-Wise Features for Amodal 3D Object Detection](https://ieeexplore.ieee.org/abstract/document/8968513)) |                                                              | IROS2019           | I+L   |
| **PI-RCNN**（[An Efficient Multi-sensor 3D Object Detector with Point-based Attentive Cont-conv Fusion Module](https://ojs.aaai.org/index.php/AAAI/article/view/6933)） |                                                              | AAAI2020           | I+L   |
| **TANet** ([TANet: Robust 3D Object Detection from Point Clouds with Triple Attention](https://arxiv.org/abs/1912.05163)) | [code](https://github.com/happinesslz/TANet)                 | AAAI2020           | L     |
| **MVF** ([End-to-end multi-view fusion for 3d object detection in lidar point clouds](https://arxiv.org/abs/1910.06528)) | [code](https://github.com/AndyYuan96/End-to-End-Multi-View-Fusion-for-3D-Object-Detection-in-LiDAR-Point-Clouds) | ICRL2020           | L     |
| **SegVoxelNet** ([SegVoxelNet: Exploring Semantic Context and Depth-aware Features for 3D Vehicle Detection from Point Cloud](https://ieeexplore.ieee.org/document/9196556)) |                                                              | ICRA2020           | L     |
| **Voxel-FPN** ([Voxel-FPN: multi-scale voxel feature aggregation in 3D object detection from point clouds](https://arxiv.org/abs/1907.05286)) |                                                              | Sensors 2020       | L     |
| **AA3D** ([Adaptive and Azimuth-Aware Fusion Network of Multimodal Local Features for 3D Object Detection](https://arxiv.org/abs/1910.04392)) |                                                              | Neurocomputing2020 | I+L   |
| **Part A^2** ([From Points to Parts: 3D Object Detection From Point Cloud With Part-Aware and Part-Aggregation Network](https://ieeexplore.ieee.org/abstract/document/9018080)) | [code](https://github.com/open-mmlab/OpenPCDet)              | TPAMI2020          | L     |
| **PV-RCNN** ([PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection](https://openaccess.thecvf.com/content_CVPR_2020/html/Shi_PV-RCNN_Point-Voxel_Feature_Set_Abstraction_for_3D_Object_Detection_CVPR_2020_paper.html)) | [code](https://github.com/open-mmlab/OpenPCDet)              | CVPR2020           | L     |
| **3D SSD** ([3DSSD: Point-based 3D Single Stage Object Detector](https://ieeexplore.ieee.org/document/9156597)) | [code](https://github.com/Jia-Research-Lab/3DSSD)            | CVPR2020           | L     |
| **Associate-3Ddet** ([Associate-3Ddet: Perceptual-to-Conceptual Association for 3D Point Cloud Object Detection](https://ieeexplore.ieee.org/document/9156880)) | [code](https://github.com/dleam/Associate-3Ddet)             | CVPR2020           | L     |
| **HVNet** ([HVNet: Hybrid Voxel Network for LiDAR Based 3D Object Detection](https://ieeexplore.ieee.org/document/9157799)) | [code](https://github.com/AndyYuan96/HVNet)                  | CVPR2020           | L     |
| **ImVoteNet** ([ImVoteNet: Boosting 3D Object Detection in Point Clouds with Image Votes](https://openaccess.thecvf.com/content_CVPR_2020/html/Qi_ImVoteNet_Boosting_3D_Object_Detection_in_Point_Clouds_With_Image_CVPR_2020_paper.html)) |                                                              | CVPR2020           | I+L   |
| **Point GNN** ([Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud](https://ieeexplore.ieee.org/document/9156733)) |                                                              | CVPR2020           | L     |
| **SA-SSD** ([Structure Aware Single-stage 3D Object Detection from Point Cloud](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Structure_Aware_Single-Stage_3D_Object_Detection_From_Point_Cloud_CVPR_2020_paper.html)) | [code](https://github.com/skyhehe123/SA-SSD)                 | CVPR2020           | L     |
| ([What You See is What You Get: Exploiting Visibility for 3D Object Detection](https://ieeexplore.ieee.org/document/9157129)) |                                                              | CVPR2020           | L     |
| **DOPS** ([DOPS: Learning to Detect 3D Objects and Predict their 3D Shapes](https://ieeexplore.ieee.org/document/9156417)) |                                                              | CVPR2020           | L     |
| **3D IoU-Net** ([3D IoU-Net: IoU Guided 3D Object Detector for Point Clouds](https://arxiv.org/abs/2004.04962)) |                                                              | Arxiv2020          | L     |
| **3D CVF** [(3D-CVF: Generating Joint Camera and LiDAR Features Using Cross-View Spatial Feature Fusion for 3D Object Detection](https://arxiv.org/abs/2004.12636)) |                                                              | ECCV2020           | I+L   |
| **HotSpotNet** ([Object as Hotspots: An Anchor-Free 3D Object Detection Approach via Firing of Hotspots](https://arxiv.org/abs/1912.12791)) |                                                              | ECCV2020           | L     |
| **EPNet**: ([EPNet: Enhancing Point Features with Image Semantics for 3D Object Detection](https://arxiv.org/abs/2007.08856)) | [code](https://github.com/happinesslz/EPNet)                 | ECCV2020           | I+L   |
| **WS3D** ([Weakly Supervised 3D Object Detection from Lidar Point Cloud](https://arxiv.org/abs/2007.11901)) | [code](https://github.com/hlesmqh/WS3D)                      | ECCV2020           | L     |
| **Pillar-OD** [Pillar-based Object Detection for Autonomous Driving](https://arxiv.org/abs/2007.10323) | [code](https://github.com/WangYueFt/pillar-od)               | ECCV2020           | L     |
| **SSN** ([SSN: Shape Signature Networks for Multi-class Object Detection from Point Clouds](https://arxiv.org/abs/2004.02774)) |                                                              | Arxiv2020          | L     |
| **CenterPoint** ([Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275)) | [code](https://github.com/tianweiy/CenterPoint)              | Arxiv2020          | L     |
| **AFDet** ([AFDet: Anchor Free One Stage 3D Object Detection](https://arxiv.org/abs/2006.12671)) |                                                              | Waymo2020          | L     |
| **LGR-Net** ([Local Grid Rendering Networks for 3D Object Detection in Point Clouds](https://arxiv.org/abs/2007.02099)) |                                                              | arxiv2020.07       | L     |
| **CenterNet3D** ([CenterNet3D:An Anchor free Object Detector for Autonomous Driving](https://arxiv.org/abs/2007.07214v3)) | [code](https://github.com/wangguojun2018/CenterNet3d)        | arxiv2020.07       | L     |
| **RCD** [(Range Conditioned Dilated Convolutions for Scale Invariant 3D Object Detection](https://arxiv.org/abs/2005.09927)) |                                                              | CoRL 2020          | L     |
| **VS3D** ([Weakly Supervised 3D Object Detection from Point Clouds](https://arxiv.org/abs/2007.13970)) | [code](https://github.com/Zengyi-Qin/Weakly-Supervised-3D-Object-Detection) | ACM MM2020         | I+L   |
|                                                              |                                                              |                    |       |
| **RangeRCNN** ([RangeRCNN: Towards Fast and Accurate 3D Object Detection with Range Image Representation](https://arxiv.org/abs/2009.00206)) |                                                              | arxiv2020.09       | L     |
| **MVAF-Net** ([Multi-View Adaptive Fusion Network for 3D Object Detection](https://arxiv.org/abs/2011.00652)) |                                                              | arxiv2020.11       | I+L   |
| **CADNet** ([Context-Aware Dynamic Feature Extraction for 3D Object Detection in  Point Clouds](http://arxiv.org/abs/1912.04775v3)) |                                                              | TITS 2021.05       | L     |
| **DA-PointRCNN** ([A Density-Aware PointRCNN for 3D Objection Detection in Point Clouds](https://arxiv.org/abs/2009.05307)) |                                                              | axiv2020.09        | L     |
| **CVCNet**(Every View Counts: Cross-View Consistency in 3D Object Detection with Hybrid-Cylindrical-Spherical Voxelization) |                                                              | NIPS2020           | L     |
| **CIA-SSD**（[CIA-SSD: Confident IoU-Aware Single-Stage Object Detector From Point Cloud](https://arxiv.org/abs/2012.03015)） | [code](https://github.com/Vegeta2020/CIA-SSD)                | AAAI2021           | L     |
| **IAAY**（[It's All Around You: Range-Guided Cylindrical Network for 3D Object Detection](https://arxiv.org/abs/2012.03121v1)） |                                                              | arxiv2020          | L     |
| **SA-Det3D** ([Self-Attention Based Context-Aware 3D Object Detection](https://arxiv.org/pdf/2101.02672.pdf)) | [code](https://github.com/AutoVision-cloud/SA-Det3D)         | arxiv2020          | L     |
| **RangeDet**([RangeDet: In Defense of Range View for LiDAR-based 3D Object Detection](https://arxiv.org/abs/2103.10039)) |                                                              | ICCV2021           | L     |
| **HVPR**([HVPR: Hybrid Voxel-Point Representation for Single-stage 3D Object Detection](https://arxiv.org/abs/2104.00902v1)) |                                                              | CVPR2021           | L     |
| **SE-SSD**([SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud](https://arxiv.org/pdf/2104.09804v1.pdf)) |                                                              | CVPR2021           | L     |
| **PPC**([To the Point: Efficient 3D Object Detection in the Range Image with Graph Convolution Kernels](https://openaccess.thecvf.com/content/CVPR2021/html/Chai_To_the_Point_Efficient_3D_Object_Detection_in_the_Range_CVPR_2021_paper.html)) |                                                              | CVPR2021           | L     |
| **SIENet** ([SIENet: Spatial Information Enhancement Network for 3D Object Detection from Point Cloud](https://arxiv.org/abs/2103.15396)) |                                                              | arxiv2021.04       | L     |
| **PolarStream**([PolarStream: Streaming Lidar Object Detection and Segmentation with Polar Pillars](https://arxiv.org/pdf/2106.07545.pdf)) |                                                              | arxiv2021.06       | L     |
| **DV-Det**([DV-Det: Efficient 3D Point Cloud Object Detection with Dynamic Voxelization](https://arxiv.org/pdf/2107.12707.pdf)) | [code](https://github.com/SUZhaoyu/dv-det)                   | arxiv2021.07       | L     |
| **VPFNet** （[VPFNet: Improving 3D Object Detection with Virtual Point based LiDAR and Stereo Data Fusion](https://arxiv.org/abs/2111.14382)） |                                                              | arxiv2021.11       | I+L   |
| **LC-MV** ([Multi-View Fusion of Sensor Data for Improved Perception and Prediction in Autonomous Driving](https://openaccess.thecvf.com/content/WACV2022/html/Fadadu_Multi-View_Fusion_of_Sensor_Data_for_Improved_Perception_and_Prediction_WACV_2022_paper.html)) |                                                              | WACV2022           | I+L   |
| **BtcDet** ([Behind the Curtain: Learning Occluded Shapes for 3D Object Detection](https://arxiv.org/abs/2112.02205)) | [code](https://github.com/Xharlie/BtcDet)                    | AAAI2022           | L     |
| To be continued...                                           |                                                              |                    |       |



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