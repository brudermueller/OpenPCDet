# OpenPCDet

`OpenPCDet` is a clear, simple, self-contained open source project for LiDAR-based 3D object detection. 

It is also the official code release of [`[PointRCNN]`](https://arxiv.org/abs/1812.04244), [`[Part-A^2 net]`](https://arxiv.org/abs/1907.03670) and [`[PV-RCNN]`](https://arxiv.org/abs/1912.13192). 


## Overview
- [Additional Documentation](#additional-documentation)
- [Changelog](#changelog)
- [Design Pattern](#openpcdet-design-pattern)
- [Model Zoo](#model-zoo)
- [Installation](docs/INSTALL.md)
- [Quick Demo](docs/DEMO.md)
- [Getting Started](docs/GETTING_STARTED.md)
- [Citation](#citation)

***
## Additional Documentation
This section gives an overview of the changes and extensions made to this toolbox for means of adjusting the _PointRCNN_ architecture to indoor environments and close-proximities. These have been developed in the context of a master project aiming at __People Detection in Close-Proximity for Robot Navigation in Crowds based on 3D LiDAR Data__ at the _Learning Algorithms and Systems Laboratory_ at EPFL. The documnentation provided by the authors on how to use this toolbox in general can be found subsequent to this section. 
### ROS Node 
For real-time inference on the robot, a ROS node can be run from the scripts: `scripts/ros_node`. It subscribes to the Velodyne LiDAR topic and feeds the received point clouds into a pre-trained model. For now, it only subscribes to one laser topic (i.e. "/front_lidar/velodyne_points"). If more, or other lasers should be taken into account, this step needs to be adapted and the point clouds would need to be merged in a pre-processing step before they are input into the network. The default values for the ROS parameters `model_path` and `detector_config` can be changed according to which pre-trained model (`.pth`-file) and which corresponding pointrcnn-config (`.yaml`-file) should be used, using `rosparam set` in the command line. Also, make sure that a master node is running. 

``` python
python3 tools/scripts/ros_node.py
```

### Visual Utils 
All code for the purpose of visualizing bounding box detections, as well as the ground truth and the point clouds themselves, can be found in `tools/visual_utils'. Two notebooks have released which can be used to interactively plot and visualize the data, also via ssh. For ssh-usage run 
```
jupyter notebook --no-browser --port 8890
```
in the `visual_utils` directory from the command line on the remote machine. The port can be also changed, if necessary. Then, on your local machine, run
```
ssh -N -f -L localhost:8890:localhost:8890 username@IP_address
```

### Data
The toolbox applicability has been extended to two additional datasets. This concerns new dataset classes, as well as evaluation and pre-processing tools. 
1. A __custom__ dataset which has been recorded from a Velodyne LiDAR VLP-16 sensor in the course of this master thesis. The corresponding source code and logic can be found in `pcdet/datasets/custom/`.
2. The __JRDB__ [dataset](https://jrdb.stanford.edu/dataset/about), which is the so far largest benchmark dataset for 2D-3D person tracking and detection. The corresponding source code and logic can be found in `pcdet/datasets/JRDB/`.

### Models 
After training/testing the models the output (`.pth`-files for the models themselves and `.txt`-files for the detections when testing) is saved to `/output`. 

The configuration for training/testing is set in yaml-config files, which can be found in `/tools/cfgs/` and they depend on the dataset to be used. Whith each train/test run a backup of the corresponding config file will also be placed in the output folder, in case the same configuration file will be changed for other models afterwards. The best performing model on the JRDB dataset, without constraining the evaluation, has been trained using `tools/cfgs/jrdb_models/pointrcnn_no_aug.yaml` and it has been saved to `/output/jrdb_models/jrdb_exp27_no_aug/`. 

For the JRDB dataset, models were also trained on a dataset which has been reduced to __indoor__ scenes only. The best performing model has been trained using `tools/cfgs/jrdb_models/pointrcnn_indoor.yaml` and has been saved to `/output/jrdb_models/jrdb_indoor_exp31/`. The evaluation/testing code has been also extended with a functionality to constrain the evaluation by a maximum distance of the ground truth objects to the sensor, as well as the minimum number of points per object point cloud. This functionality can be used by adding `--ignore_hard_class` as an argument on the command line.  

### Useful example commands to get started 
- Create pickle-files to initialize a new dataset (only necessary once, if you want to change something in the data input)
  ```
  python3 -m pcdet.datasets.JRDB.jrdb_dataset create_jrdb_infos tools/cfgs/dataset_configs/jrdb_dataset.yaml
  ```
- Train a model with multiple GPUs (in this example with 2 GPUS and the weights of a pre-trained model): 
  ```
  bash scripts/dist_train.sh 2 --cfg_file cfgs/jrdb_models/pointrcnn.yaml --extra_tag jrdb_exp_xy --ckpt_save_interval 1 --pretrained_model ../output/custom_models/pointrcnn/l1_corner_loss_80epochs/ckpt/checkpoint_epoch_80.pth  
  ```

- Train with one GPU only: 
  ```
  python3 train.py --cfg_file cfgs/jrdb_models/pointrcnn_no_pretrained.yaml --extra_tag jrdb_indoor_exp20 --ckpt_save_interval 1
  ```

- Test/Validate a model with multiple GPUs: 
  ``` 
  bash scripts/dist_test.sh 2 --cfg_file cfgs/jrdb_models/pointrcnn_angle_loss.yaml --batch_size 4 --eval_tag jrdb --extra_tag jrdb30_angle_loss  --ckpt ../output/jrdb_models/jrdb_exp30_angle_loss/ckpt/checkpoint_epoch_30.pth --save_to_file
  ```

***
## Changelog
[2021-01-10] Add support for two new indoor datasets, extend visualization tools, add real-time ROS node, add model configurations for indoor/crowd-navigation.

[2020-11-10] **NEW:** The [Waymo Open Dataset](#waymo-open-dataset-baselines) has been supported with state-of-the-art results. Currently we provide the 
configs and results of `SECOND`, `PartA2` and `PV-RCNN` on the Waymo Open Dataset, and more models could be easily supported by modifying their dataset configs. 

[2020-08-10] Bugfixed: The provided NuScenes models have been updated to fix the loading bugs. Please redownload it if you need to use the pretrained NuScenes models.

[2020-07-30] `OpenPCDet` v0.3.0 is released with the following features:
   * The Point-based and Anchor-Free models ([`PointRCNN`](#KITTI-3D-Object-Detection-Baselines), [`PartA2-Free`](#KITTI-3D-Object-Detection-Baselines)) are supported now.
   * The NuScenes dataset is supported with strong baseline results ([`SECOND-MultiHead (CBGS)`](#NuScenes-3D-Object-Detection-Baselines) and [`PointPillar-MultiHead`](#NuScenes-3D-Object-Detection-Baselines)).
   * High efficiency than last version, support **PyTorch 1.1~1.7** and **spconv 1.0~1.2** simultaneously.
   
[2020-07-17]  Add simple visualization codes and a quick demo to test with custom data. 

[2020-06-24] `OpenPCDet` v0.2.0 is released with pretty new structures to support more models and datasets. 

[2020-03-16] `OpenPCDet` v0.1.0 is released. 


## Introduction


### What does `OpenPCDet` toolbox do?

Note that we have upgrated `PCDet` from `v0.1` to `v0.2` with pretty new structures to support various datasets and models.

`OpenPCDet` is a general PyTorch-based codebase for 3D object detection from point cloud. 
It currently supports multiple state-of-the-art 3D object detection methods with highly refactored codes for both one-stage and two-stage 3D detection frameworks.

Based on `OpenPCDet` toolbox, we win the Waymo Open Dataset challenge in [3D Detection](https://waymo.com/open/challenges/3d-detection/), 
[3D Tracking](https://waymo.com/open/challenges/3d-tracking/), [Domain Adaptation](https://waymo.com/open/challenges/domain-adaptation/) 
three tracks among all LiDAR-only methods, and the Waymo related models will be released to `OpenPCDet` soon.    

We are actively updating this repo currently, and more datasets and models will be supported soon. 
Contributions are also welcomed. 

### `OpenPCDet` design pattern

* Data-Model separation with unified point cloud coordinate for easily extending to custom datasets:
<p align="center">
  <img src="docs/dataset_vs_model.png" width="95%" height="320">
</p>

* Unified 3D box definition: (x, y, z, dx, dy, dz, heading).

* Flexible and clear model structure to easily support various 3D detection models: 
<p align="center">
  <img src="docs/model_framework.png" width="95%">
</p>

* Support various models within one framework as: 
<p align="center">
  <img src="docs/multiple_models_demo.png" width="95%">
</p>


### Currently Supported Features

- [x] Support both one-stage and two-stage 3D object detection frameworks
- [x] Support distributed training & testing with multiple GPUs and multiple machines
- [x] Support multiple heads on different scales to detect different classes
- [x] Support stacked version set abstraction to encode various number of points in different scenes
- [x] Support Adaptive Training Sample Selection (ATSS) for target assignment
- [x] Support RoI-aware point cloud pooling & RoI-grid point cloud pooling
- [x] Support GPU version 3D IoU calculation and rotated NMS 


## Model Zoo

### KITTI 3D Object Detection Baselines
Selected supported methods are shown in the below table. The results are the 3D detection performance of moderate difficulty on the *val* set of KITTI dataset.
* All models are trained with 8 GTX 1080Ti GPUs and are available for download. 
* The training time is measured with 8 TITAN XP GPUs and PyTorch 1.5.

|                                             | training time | Car@R11 | Pedestrian@R11 | Cyclist@R11  | download | 
|---------------------------------------------|----------:|:-------:|:-------:|:-------:|:---------:|
| [PointPillar](tools/cfgs/kitti_models/pointpillar.yaml) |~1.2 hours| 77.28 | 52.29 | 62.68 | [model-18M](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing) | 
| [SECOND](tools/cfgs/kitti_models/second.yaml)       |  ~1.7 hours  | 78.62 | 52.98 | 67.15 | [model-20M](https://drive.google.com/file/d/1-01zsPOsqanZQqIIyy7FpNXStL3y4jdR/view?usp=sharing) |
| [PointRCNN](tools/cfgs/kitti_models/pointrcnn.yaml) | ~3 hours | 78.70 | 54.41 | 72.11 | [model-16M](https://drive.google.com/file/d/1BCX9wMn-GYAfSOPpyxf6Iv6fc0qKLSiU/view?usp=sharing)| 
| [PointRCNN-IoU](tools/cfgs/kitti_models/pointrcnn_iou.yaml) | ~3 hours | 78.75 | 58.32 | 71.34 | [model-16M](https://drive.google.com/file/d/1V0vNZ3lAHpEEt0MlT80eL2f41K2tHm_D/view?usp=sharing)|
| [Part-A^2-Free](tools/cfgs/kitti_models/PartA2_free.yaml)   | ~3.8 hours| 78.72 | 65.99 | 74.29 | [model-226M](https://drive.google.com/file/d/1lcUUxF8mJgZ_e-tZhP1XNQtTBuC-R0zr/view?usp=sharing) |
| [Part-A^2-Anchor](tools/cfgs/kitti_models/PartA2.yaml)    | ~4.3 hours| 79.40 | 60.05 | 69.90 | [model-244M](https://drive.google.com/file/d/10GK1aCkLqxGNeX3lVu8cLZyE0G8002hY/view?usp=sharing) |
| [PV-RCNN](tools/cfgs/kitti_models/pv_rcnn.yaml) | ~5 hours| 83.61 | 57.90 | 70.47 | [model-50M](https://drive.google.com/file/d/1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-/view?usp=sharing) |

### NuScenes 3D Object Detection Baselines
All models are trained with 8 GTX 1080Ti GPUs and are available for download.

|                                             | mATE | mASE | mAOE | mAVE | mAAE | mAP | NDS | download | 
|---------------------------------------------|----------:|:-------:|:-------:|:-------:|:---------:|:-------:|:-------:|:---------:|
| [PointPillar-MultiHead](tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml) | 33.87	| 26.00 | 32.07	| 28.74 | 20.15 | 44.63 | 58.23	 | [model-23M](https://drive.google.com/file/d/1p-501mTWsq0G9RzroTWSXreIMyTUUpBM/view?usp=sharing) | 
| [SECOND-MultiHead (CBGS)](tools/cfgs/nuscenes_models/cbgs_second_multihead.yaml) | 31.15 |	25.51 |	26.64 | 26.26 | 20.46 | 50.59 | 62.29 | [model-35M](https://drive.google.com/file/d/1bNzcOnE3u9iooBFMk2xK7HqhdeQ_nwTq/view?usp=sharing) |

### Waymo Open Dataset Baselines
We provide the setting of [`DATA_CONFIG.SAMPLED_INTERVAL`](tools/cfgs/dataset_configs/waymo_dataset.yaml) on the Waymo Open Dataset (WOD) to subsample partial samples for training and evaluation, 
so you could also play with WOD by setting a smaller `DATA_CONFIG.SAMPLED_INTERVAL` even if you only have limited GPU resources. 

By default, all models are trained with **20% data (~32k frames)** of all the training samples on 8 GTX 1080Ti GPUs, and the results of each cell here are mAP/mAPH calculated by the official Waymo evaluation metrics on the **whole** validation set (version 1.2).    

|                                             | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |  
|---------------------------------------------|----------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| [SECOND](tools/cfgs/waymo_models/second.yaml) | 68.03/67.44	| 59.57/59.04 | 61.14/50.33	| 53.00/43.56 | 54.66/53.31 | 52.67/51.37 | 
| [Part-A^2-Anchor](tools/cfgs/waymo_models/PartA2.yaml) | 71.82/71.29 | 64.33/63.82 | 63.15/54.96 | 54.24/47.11 | 65.23/63.92 | 62.61/61.35 |
| [PV-RCNN](tools/cfgs/waymo_models/pv_rcnn.yaml) | 74.06/73.38 | 64.99/64.38 |	62.66/52.68 | 53.80/45.14 |	63.32/61.71	| 60.72/59.18 | 

We could not provide the above pretrained models due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/), 
but you could easily achieve similar performance by training with the default configs.



### Other datasets
More datasets are on the way. 

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of `OpenPCDet`.


## Quick Demo
Please refer to [DEMO.md](docs/DEMO.md) for a quick demo to test with a pretrained model and 
visualize the predicted results on your custom data or the original KITTI data.

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.


## License

`OpenPCDet` is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
`OpenPCDet` is an open source project for LiDAR-based 3D scene perception that supports multiple
LiDAR-based perception models as shown above. Some parts of `PCDet` are learned from the official released codes of the above supported methods. 
We would like to thank for their proposed methods and the official implementation.   

We hope that this repo could serve as a strong and flexible codebase to benefit the research community by speeding up the process of reimplementing previous works and/or developing new methods.


## Citation 
If you find this project useful in your research, please consider cite:


```
@misc{openpcdet2020,
    title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
    author={OpenPCDet Development Team},
    howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}},
    year={2020}
}
```

## Contribution
Welcome to be a member of the OpenPCDet development team by contributing to this repo, and feel free to contact us for any potential contributions. 


