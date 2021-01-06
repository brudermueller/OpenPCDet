import collections
import copy
import json
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import (box_utils, common_utils, custom_data_utils, jrdb_utils,
                      object3d_jrdb, plot_utils)
from ..dataset import DatasetTemplate


class JrdbDataset(DatasetTemplate): 
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None): 
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        # self.root = os.path.join(root, 'custom_data')
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.data_path = self.dataset_cfg.DATA_ROOT

        print('Root Path: {}'.format(self.root_path))
        print('Data Path: {}'.format(self.data_path))
        self.split_dir = os.path.join(self.root_path, self.split + '.txt')
        if self.split in ['train','val']:
            label_split = 'train'
            self.label_dir = os.path.join(self.data_path, f'{label_split}_dataset/labels/labels_3d')
        else: 
            label_split = 'test'
            self.label_dir = None 

        self.image_path = os.path.join(self.data_path, f'{label_split}_dataset/images/image_stitched')

        if self.logger is not None: 
            self.logger.info('Load samples from %s' % self.split_dir)

        # Create Mapping from sample frames to frame ids 
        self.current_samples = [x.strip() for x in open(self.split_dir).readlines()] if os.path.exists(self.split_dir) else None
        self.sample_id_list = [idx for idx in range(0, self.current_samples.__len__())]
        self.sample_id_map = {name:idx for (idx, name) in enumerate(self.current_samples)}
        self.num_sample = self.sample_id_list.__len__()
        
        self.label_dict = {}
        self.data_infos = []
        self.include_data(self.mode)
        if training: 
            self.get_all_labels()

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.data_infos) * self.total_epochs

        return len(self.data_infos)


    def __getitem__(self, index): 
        """
        Function to load the raw data (and labels) and call the function self.prepare_data() to 
        process the data and send it to the model.

        Args:
            index 
        """
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.data_infos)

        info = copy.deepcopy(self.data_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']

        points = self.get_lidar(sample_idx)

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
        }
        # print(
        #     'Sample id: {} \nPoints: {}'.format(sample_idx, points.shape)
        # )

        if 'annos' in info:
            annos = info['annos']
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar,
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict 
    
    def get_image(self,index): 
        frame = self.current_samples[index]
        file_parts = frame.split('/')
        sequence = file_parts[len(file_parts)-2]   
        frame_num = file_parts[-1].split('.')[0]
        return jrdb_utils.load_image(os.path.join(self.image_path, sequence, frame_num+'.jpg')) 

    def include_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading JRDB dataset')
        data_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                data_infos.extend(infos)

        self.data_infos.extend(data_infos)

        if self.logger is not None:
            self.logger.info('Total samples for JRDB dataset: %d' % (len(data_infos)))

    def get_all_labels(self): 
        label_dict = collections.defaultdict(dict)   
        for json_file in os.listdir(self.label_dir): 
            sequence = os.path.basename(os.path.splitext(json_file)[0])
            file_dict = custom_data_utils.load_json_file(os.path.join(self.label_dir, json_file))
            for frame, annotations_list in file_dict['labels'].items(): 
                frame_num = frame.split('.')[0]
                box_list = []
                for anno in annotations_list:                   
                    box_dict = anno['box']
                    xyz = np.array([box_dict["cx"], box_dict["cy"], box_dict["cz"]], dtype=np.float32).reshape(1, 3)
                    lwh = np.array([box_dict["l"], box_dict["w"], box_dict["h"]], dtype=np.float32).reshape(1, 3)
                    box = np.concatenate([xyz, lwh, np.reshape(box_dict['rot_z'], (-1,1))], axis=1)
                    box_list.append(box)
                label_dict[sequence][frame_num] = box_list 
        print('Loaded all labels for JRDB')
        self.label_dict = label_dict

    def get_all_ignore_labels(self): 
        label_dict = collections.defaultdict(dict)
        for json_file in os.listdir(self.label_dir): 
            sequence = os.path.basename(os.path.splitext(json_file)[0])
            file_dict = custom_data_utils.load_json_file(os.path.join(self.label_dir, json_file))
            for frame, annotations_list in file_dict['labels'].items(): 
                ignore_mask = np.zeros(len(annotations_list), dtype=bool)
                frame_num = frame.split('.')[0]
                for i, anno in enumerate(annotations_list): 
                    dist = anno['attributes']['distance']
                    num_points = anno['attributes']['num_points']
                    if num_points < 5 or dist > 7:
                        ignore_mask[i] = 1
                label_dict[sequence][frame_num] = ignore_mask 
        return label_dict
    
    def get_ignore_mask(self, index, all_ignore_labels):
        frame = self.current_samples[index]
        file_parts = frame.split('/')
        sequence = file_parts[len(file_parts)-2]   
        frame_num = file_parts[-1].split('.')[0]
        ignore_mask = all_ignore_labels[sequence][frame_num]
        return ignore_mask

    def get_label(self, index):
        """ 
        Return bbox annotations per frame, defined as (N,7), i.e. (N x [x, y, z, h, w, l, ry])
        
        Args:
            index 
        """
        frame = self.current_samples[index]
        file_parts = frame.split('/')
        sequence = file_parts[len(file_parts)-2]   
        frame_num = file_parts[-1].split('.')[0]
        try: 
            bbox_list = np.reshape(self.label_dict[sequence][frame_num],(-1,7)) 
        except KeyError: 
            print("Label dict does not contain key combination (seq: {}, frame_num: {})".format(sequence, frame_num))
            print(label)
        bbox_obj_list = [object3d_jrdb.Object3d(box, gt=True) for box in bbox_list]
        return bbox_obj_list

    def get_lidar(self, index):        
        """ 
        Return lidar point data loaded from h5 files from lower and upper velodyne transformed in common base frame in form of (N,4).

        Args:
            index 
        """
        frame = self.current_samples[index]
        lidar_file_upper = self.data_path + frame
        lidar_file_lower = self.data_path + frame.replace('upper', 'lower')
        assert os.path.exists(lidar_file_upper), 'Lidar file "{}" does not exist.'.format(lidar_file_upper)
        assert os.path.exists(lidar_file_lower), 'Lidar file "{}" does not exist.'.format(lidar_file_lower)
        
        pts_upper = np.array(custom_data_utils.load_h5_basic(lidar_file_upper), dtype=np.float32)
        pts_upper[:,:3] = jrdb_utils.transform_pts_upper_velodyne_to_base(pts_upper[:, 0:3].T).T
        pts_lower = np.array(custom_data_utils.load_h5_basic(lidar_file_lower), dtype=np.float32)
        pts_lower[:,:3] = jrdb_utils.transform_pts_lower_velodyne_to_base(pts_lower[:, 0:3].T).T

        # scale intensity value 
        if self.dataset_cfg.POINT_FEATURE_ENCODING.scale_intensity: 
            pts_upper[:,3] = minmax_scale(pts_upper[:,3], axis=0)
            pts_lower[:,3] = minmax_scale(pts_lower[:,3], axis=0)

        pts_merged = np.concatenate((pts_upper, pts_lower), axis=0)
        return pts_merged

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples),
                'alpha': np.zeros(num_samples),
                'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            # save boxes also in camera coordinates for kitti eval 
            pred_boxes_camera = box_utils.boxes3d_lidar_to_cam_custom(pred_boxes)
            beta = np.arctan2(pred_boxes_camera[:,1], pred_boxes_camera[:,0])
            alpha = -np.sign(beta) * np.pi / 2 + beta + pred_boxes_camera[:, 6]
            pred_dict['alpha'] = alpha # -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    pred_box_lidar = single_pred_dict['boxes_lidar']
                    # print(pred_box_lidar.shape)
                    loc = pred_box_lidar[:, 0:3]
                    dims = pred_box_lidar[:, 3:6] # lwh -> hwl

                    for idx in range(len(pred_box_lidar)):
                        print('%s %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], # class
                                 single_pred_dict['alpha'][idx], # alpha 
                                 loc[idx][0],loc[idx][1], loc[idx][2], # x, y, z 
                                 dims[idx][2], dims[idx][1], dims[idx][0], # h, w, l 
                                 single_pred_dict['rotation_y'][idx], # angle 
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['dimensions'] = np.array([[obj.l, obj.w, obj.h] for obj in obj_list])  # lwh (lidar) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])

                num_objects = len([obj.cls_type for obj in obj_list])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc_lidar = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                # loc_lidar = calib.rect_to_lidar(loc)
                l, w, h = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                # loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots[..., np.newaxis]], axis=1)
                # print('Gt boxes lidar: {}'.format(np.mean(gt_boxes_lidar, axis=0)))
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                annotations['gt_boxes_camera'] = box_utils.boxes3d_lidar_to_cam_custom(gt_boxes_lidar)
                # print('Gt boxes camera: {}'.format(np.mean(annotations['gt_boxes_camera'], axis=0)))
                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    # print('got_lidar')
                    # image = self.get_image(sample_idx)
                    # print('got_image')
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar, rot_mat_alt=self.dataset_cfg.ROT_MAT_ALT)
                    # plot_utils.plot_dets_bev(points,  gt_boxes_lidar, frame=sample_idx, image=image, save_fig=True, extra_tag='sanity_check_')
                    # print('plot_utils_2')

                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(points[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt
            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
            import torch

            database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
            db_info_save_path = Path(self.root_path) / ('jrdb_dbinfos_%s.pkl' % split)

            database_save_path.mkdir(parents=True, exist_ok=True)
            all_db_infos = {}
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)

            for k in range(len(infos)):
            # for k in range(1):
                print('gt_database sample: %d/%d' % (k + 1, len(infos)))
                info = infos[k]
                sample_idx = info['point_cloud']['lidar_idx']
                points = np.array(self.get_lidar(sample_idx),dtype=np.float32)
                annos = info['annos']
                names = annos['name']
                # bbox = annos['bbox']
                gt_boxes = annos['gt_boxes_lidar']

                num_obj = gt_boxes.shape[0]
                point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
                ).numpy()  # (nboxes, npoints)

                for i in range(num_obj):
                    # print('---------- Sample {}: object {}----------'.format(sample_idx, i))
                    # filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                    filename = '%s_%s_%d.h5' % (sample_idx, names[i], i)
                    filepath = database_save_path / filename
                    gt_points = points[point_indices[i] > 0]
                    # print('num_points_in_gt: {}'.format(gt_points.shape[0]))

                    gt_points[:, :3] -= gt_boxes[i, :3] # substract box center from filtered points within box
                    # print(gt_points)

                    if np.isinf(gt_points).any() or np.isnan(gt_points).any(): 
                        import pdb 
                        pdb.set_trace()

                    if not gt_points.shape[0] == 0: 
                        # print('Saving {} to {}'.format(gt_points.shape, filepath))
                        custom_data_utils.save_h5_basic(filepath, gt_points)
                        # with open(filepath, 'w') as f:
                        #     gt_points.tofile(f)
                    else: 
                        continue
                    if (used_classes is None) or names[i] in used_classes:
                        db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                        db_info = {'name': names[i], 'path': db_path, 'sample_idx': sample_idx, 'gt_idx': i,
                                'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                                'score': annos['score'][i]}
                        if names[i] in all_db_infos:
                            all_db_infos[names[i]].append(db_info)
                        else:
                            all_db_infos[names[i]] = [db_info]
            for k, v in all_db_infos.items():
                print('Database %s: %d' % (k, len(v)))

            with open(db_info_save_path, 'wb') as f:
                pickle.dump(all_db_infos, f)

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split

        split_dir = self.root_path / (self.split + '.txt')
        self.current_samples = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(self.split_dir) else None
        self.sample_id_list = [idx for idx in range(0, self.current_samples.__len__())]
        self.sample_id_map = {name:idx for (idx, name) in enumerate(self.current_samples)}

    def evaluation(self, det_annos, class_names, **kwargs): 
        if 'annos' not in self.data_infos[0].keys():
            return None, {}
        output_path = Path(kwargs['output_path'])
        easy_eval = kwargs['easy_eval']
        print(output_path)

        from ..custom.eval import evaluate as custom_eval
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.data_infos]

        tp_mask_total = collections.defaultdict(list)
        angular_similarity_iou = collections.defaultdict(list)
        iou_thresholds = np.arange(0.3, 0.9, 0.2)
        frames_no_dets = []

        if easy_eval: 
            all_ignore_labels = self.get_all_ignore_labels()
        # calculate true positive mask for each frame and each iou threshold
        all_gt_boxes = np.empty([0,7])
        all_dt_boxes = np.empty([0,7])
        all_scores = np.empty([0])

        for dt in tqdm(eval_det_annos):             
            frame_id = dt['frame_id']
            # print('--------- {} --------- '.format(frame_id))
            gt_boxes = self.data_infos[frame_id]['annos']['gt_boxes_lidar']
            dt_boxes = dt['boxes_lidar']
            scores = dt['score']
            num_dets = dt_boxes.shape[0]
            if easy_eval:
                ignore_mask = self.get_ignore_mask(frame_id, all_ignore_labels)
                # print('Ignoring {} objects for evaluation. '.format(np.sum(~ignore_mask)))
                gt_boxes = self.data_infos[frame_id]['annos']['gt_boxes_lidar'][~ignore_mask, :]
                if gt_boxes.shape[0]== 0: 
                    continue
                # also ignore detections further away than distance threshold 
                dt_boxes = dt['boxes_lidar']
                center_depth = np.linalg.norm(dt_boxes[:, 0:3], axis=1)
                det_include_idx = np.where(center_depth < 7)[0]
                dt_boxes = dt_boxes[det_include_idx, :]
                
                scores = dt['score'][det_include_idx]
                num_dets = dt_boxes.shape[0]

            all_gt_boxes = np.concatenate((all_gt_boxes, gt_boxes),0)
            all_dt_boxes = np.concatenate((all_dt_boxes, dt_boxes),0) 
            all_scores = np.concatenate((all_scores, scores),0)

            if num_dets == 0: 
                frames_no_dets.append(frame_id) 
                continue    
            # calculate overlaps for different iou thresholds 
            # separate calculation due to limited GPU memory
            tp_mask, gt_indices = custom_eval.get_tp_mask_from_overlaps(dt_boxes, gt_boxes, iou_thresholds)
            for iou, mask in tp_mask.items(): 
                tp_mask_total[iou].append(mask)
                # calculate similarity measure from gt_indices 
                angular_similarity_list = []
                for i in range(dt_boxes.shape[0]):
                    angular_similarity_list.append((1+ np.cos(dt_boxes[i, 6] - gt_boxes[gt_indices[i],6]))/2 * mask[i])
                angular_similarity_iou[iou].append(angular_similarity_list)
            
        # concatenate all true positive masks over all frame ids for all iou thresholds 
        tp_mask_concat = {iou: np.concatenate(mask) for iou, mask in tp_mask_total.items()}
        angular_dict_concat =  {iou: np.concatenate(ang_sim) for iou, ang_sim in angular_similarity_iou.items()}
        # gt_boxes = np.concatenate([gt['gt_boxes_lidar'] for gt in eval_gt_annos],0)
        # dt_boxes = np.concatenate([dt['boxes_lidar'] for dt in eval_det_annos],0)
        # scores = np.concatenate([dt['score'] for dt in eval_det_annos],0)

        # result_str, result_dict, pr_dict = custom_eval.get_results_distributed(
        #     dt_boxes, gt_boxes, scores, tp_mask_concat, angular_dict_concat, output_path)

        result_str, result_dict = custom_eval.get_results_distributed(
            all_dt_boxes, all_gt_boxes, all_scores, tp_mask_concat, angular_dict_concat, output_path)

        print('Frames without detections: {}'.format(len(frames_no_dets)))
        with open(f"{output_path}/../frames_without_dets.txt", "w") as output:
            for frame in frames_no_dets:
                s = " ".join(map(str, str(frame)))
                output.write(s+'\n')
        return result_str, result_dict


def create_jrdb_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = JrdbDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    dataset.get_all_labels()
    # import pdb
    # pdb.set_trace()
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('jrdb_infos_%s.pkl' % train_split)
    val_filename = save_path / ('jrdb_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'jrdb_infos_trainval.pkl'
    test_filename = save_path / 'jrdb_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    jrdb_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(jrdb_infos_train, f)
    print('JRDB info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    jrdb_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(jrdb_infos_val, f)
    print('JRDB info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(jrdb_infos_train + jrdb_infos_val, f)
    print('JRDB info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    jrdb_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(jrdb_infos_test, f)
    print('JRDB info test file is saved to %s' % test_filename)


    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_jrdb_infos':
        from pathlib import Path
        import yaml
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_jrdb_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Pedestrian'],
            # data_path=ROOT_DIR / 'data' / 'jrdb_temp',
            # save_path=ROOT_DIR / 'data' / 'jrdb_temp'
            data_path=ROOT_DIR / 'data' / 'jrdb_indoor',
            save_path=ROOT_DIR / 'data' / 'jrdb_indoor'
        )
