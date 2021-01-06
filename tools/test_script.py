import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

import pdb
from pathlib import Path

import numpy as np
from pcdet.config import (cfg, cfg_from_list, cfg_from_yaml_file,
                          log_config_to_file)
from pcdet.datasets.JRDB.jrdb_dataset import JrdbDataset
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.utils import plot_utils

# root_path = Path('data/jrdb')

cfg_from_yaml_file('cfgs/kitti_models/pointrcnn_test.yaml', cfg)

dataset = JrdbDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=['Pedestrian'],
        root_path=None,
        training=True,
        logger=None,
    )
# print(dataset.sample_id_list)
# dataset.get_infos(sample_id_list=dataset.sample_id_list)


# cfg_from_yaml_file('cfgs/kitti_models/pointrcnn.yaml', cfg)

# dataset = KittiDataset(
#         dataset_cfg=cfg.DATA_CONFIG,
#         class_names=['Pedestrian'],
#         root_path=None,
#         training=True,
#         logger=None,
#     )

num_frames = len(dataset.data_infos)
# num_frames = len(dataset.kitti_infos)

# for i in range(100):
#     print(dataset.__getitem__(i))
#     sampled_points = dataset.__getitem__(i)['points']
#     i = dataset.kitti_infos[i]['point_cloud']['lidar_idx']
#     orginal_points = dataset.get_lidar(i)
#     print(np.mean(orginal_points, axis=0) - np.mean(sampled_points, axis=0))

# for i in range(100):
for i in range(num_frames):
    sampled_points = dataset.__getitem__(i)['points']
    new_boxes = dataset.__getitem__(i)['gt_boxes']
    orginal_points = dataset.get_lidar(i)
    image = dataset.get_image(i)
    obj_list= dataset.get_label(i)
    # print(np.mean(orginal_points, axis=0) - np.mean(sampled_points, axis=0))

    dims = np.array([[obj.l, obj.w, obj.h] for obj in obj_list])  # lwh (lidar) format
    l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar= np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
    rots= np.array([obj.ry for obj in obj_list])

    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots[..., np.newaxis]], axis=1)
    # plot_utils.plot_dets_bev(orginal_points,  gt_boxes_lidar, frame=i, image=image, save_fig=True, extra_tag='sanity_check/sanity_check_', extra_points=sampled_points)
    plot_utils.plot_dets_gt_bev(orginal_points,  gt_boxes_lidar, new_boxes, frame=i, image=image, extra_points= sampled_points, save_fig=True, extra_tag='sanity_check/sampling_',)


# pdb.set_trace()
