""" 
Taken from: https://raw.githubusercontent.com/StanfordVL/JRMOT_ROS/master/src/3d_detector.py
TODO: to be adapted to our purpose, just serves as inspiration 
"""
import os
import pdb
import sys
import time

import cv2
import message_filters
import numpy as np
import ros_numpy
import rospy
import sensor_msgs.point_cloud2 as pc2
import torch
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, Vector3
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from pcdet.config import (cfg, cfg_from_list, cfg_from_yaml_file,
                          log_config_to_file)
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pyquaternion import Quaternion
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header, Int8
from visualization_msgs.msg import Marker, MarkerArray


def yaw2quaternion(yaw: float) -> Quaternion:
    return Quaternion(axis=[0,0,1], radians=yaw)

class DummyDataset(DatasetTemplate): 
    def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None):
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
        self.root_path = root_path
        self.sample_file_list = []

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        points = self.sample_file_list[index]

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


class Detector_3d:
    def __init__(self):
        self.node_name = "detector"
        
        rospy.init_node(self.node_name)
        rospy.on_shutdown(self.cleanup)

        self.device = None
        self.net = None 

        
        self.detector_config = \
            rospy.get_param('~model_path',
                            '../../models/ros_node_model/pointrcnn.yaml')
        
        self.model_path = \
            rospy.get_param('~detector_config',
                            '../../models/ros_node_model/checkpoint_epoch_80.pth')
        

        # load model config 
        self.setup_model()
        sub_ = rospy.Subscriber("/front_lidar/velodyne_points", PointCloud2, self.detector_callback, queue_size=1, buff_size=2**24)
        self.pub_arr_bbox = rospy.Publisher("/detection_markers", BoundingBoxArray, queue_size=1)

        self.inference_times = []

        rospy.loginfo("3D detector ready.")


    def setup_model(self): 
        cfg_from_yaml_file(self.detector_config, cfg)
        self.logger = common_utils.create_logger()
        self.dataset = DummyDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.dataset)
        self.net.load_params_from_file(filename=self.model_path, logger=self.logger, to_cpu=True)
        self.net = self.net.to(self.device).eval()

    def run_model(self, points):
        t_t = time.time()
        # rospy.loginfo('Input: pointcloud with shape {}'.format(points.shape))
        input_dict = {
            'points': points,
            'frame_id': 0,
        }

        data_dict = self.dataset.prepare_data(data_dict=input_dict)
        data_dict = self.dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        torch.cuda.synchronize()
        t = time.time()

        # pred_dicts, _ = self.net.forward(data_dict)
        with torch.no_grad():
            pred_dicts, _ = self.net(data_dict)
        
        torch.cuda.synchronize()
        inference_time = time.time() - t
        self.inference_times.append(inference_time)
        rospy.loginfo(f" PointRCNN inference cost time: {time.time() - t}")
        rospy.loginfo("Stdev: {}".format(np.std(self.inference_times)))

        boxes_lidar = pred_dicts[0]["pred_boxes"].detach().cpu().numpy()
        scores = pred_dicts[0]["pred_scores"].detach().cpu().numpy()
        types = pred_dicts[0]["pred_labels"].detach().cpu().numpy()
        # rospy.loginfo('Detected {} persons.'.format(boxes_lidar.shape[0]))

        return scores, boxes_lidar, types

        
    def detector_callback(self, pcl_msg):
        start = time.time()
        # rospy.loginfo('Processing Pointcloud with PointRCNN')
        arr_bbox = BoundingBoxArray()
        seq = pcl_msg.header.seq
        stamp = pcl_msg.header.stamp
        # in message pointcloud has x pointing forward, y pointing to the left and z pointing upward
        pts_lidar = np.array([[p[0], p[1], p[2],p[3]] for p in pc2.read_points(pcl_msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))], dtype=np.float32)
        scores, dt_box_lidar, types = self.run_model(pts_lidar)


        # TODO: question convert into torch tensors? torch.from_numpy(pts_lidar)
        # move onto gpu if available
  
        # TODO: check if needs translation/rotation to compensate for tilt etc. 
        
        if scores.size != 0:
            for i in range(scores.size):
                bbox = BoundingBox()
                bbox.header.frame_id = pcl_msg.header.frame_id
                bbox.header.stamp = rospy.Time.now()
                # bbox.header.seq = pcl_msg.header.seq
                q = yaw2quaternion(float(dt_box_lidar[i][6]))
                bbox.pose.orientation.x = q[1]
                bbox.pose.orientation.y = q[2]
                bbox.pose.orientation.z = q[3]
                bbox.pose.orientation.w = q[0]           
                bbox.pose.position.x = float(dt_box_lidar[i][0])
                bbox.pose.position.y = float(dt_box_lidar[i][1])
                bbox.pose.position.z = float(dt_box_lidar[i][2])
                bbox.dimensions.x = float(dt_box_lidar[i][3])
                bbox.dimensions.y = float(dt_box_lidar[i][4])
                bbox.dimensions.z = float(dt_box_lidar[i][5])
                bbox.value = scores[i]
                bbox.label = int(types[i])
                arr_bbox.boxes.append(bbox)

        # rospy.loginfo("3D detector time: {}".format(time.time() - start))
        
        arr_bbox.header.frame_id = pcl_msg.header.frame_id
        arr_bbox.header.stamp = pcl_msg.header.stamp
        arr_bbox.header.seq = pcl_msg.header.seq

        
        if len(arr_bbox.boxes) != 0:
            self.pub_arr_bbox.publish(arr_bbox)
            arr_bbox.boxes = []
        else:
            arr_bbox.boxes = []
            self.pub_arr_bbox.publish(arr_bbox)
        


    def cleanup(self):
        print("Shutting down 3D-Detection node.")
    
def main(args):       
    try:
        Detector_3d()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down 3D-Detection node.")
        print(np.std(Detector_3d.inference_times))

if __name__ == '__main__':
    main(sys.argv)
