from pcdet.datasets.JRDB.jrdb_dataset import JrdbDataset
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pathlib import Path

import pdb
import numpy as np

# root_path = Path('data/jrdb')

cfg_from_yaml_file('cfgs/kitti_models/pointrcnn_test.yaml', cfg)

dataset = JrdbDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=['Pedestrian'],
        root_path=None,
        training=True,
        logger=None,
    )

dataset.set_split('train')
train_filename = (Path(__file__).resolve() / '../../').resolve() / 'data' / 'jrdb_temp' / ('jrdb_infos_train.pkl')

dataset.create_groundtruth_database(train_filename, split='train')


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
#     # print(dataset.__getitem__(i))
#     sampled_points = dataset.__getitem__(i)['points']
#     orginal_points = dataset.get_lidar(i)
#     print(np.mean(orginal_points, axis=0) - np.mean(sampled_points, axis=0))



# pdb.set_trace()