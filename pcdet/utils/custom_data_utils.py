import os 
import sys 
import numpy as np 
import scipy.io as sio
import h5py
import json

from scipy.spatial.transform import Rotation as R


'''
    Helper functions to write .h5 data files for pointnet, etc.
'''

def pts_lidar_to_camera(pts_lidar): 
    """ Projects points into camera coordinate frame (x=right, y=down, z=forward) 
        from LiDAR Velodyne frame (x=forward, y=left, z=up). 

    Args:
        pts (numpy nd array (N,3)): [description]
    """
    rot = R.from_euler('yx', [90, -90], degrees=True)
    rot_mat = rot.as_matrix().T
    pts_cam = np.dot(rot_mat, pts_lidar.T)
    return pts_cam.T


def save_h5_basic(h5_filename, data, data_dtype='float32'):
    h5_fout = h5py.File(h5_filename, 'w')
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype,
    )
    h5_fout.close()


def save_h5(h5_filename, data, label, bbox=None, data_dtype='float32', label_dtype='int', 
            bbox_dtype='float32'):
    h5_fout = h5py.File(h5_filename, 'w')
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype,
    )
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype,
    )
    if isinstance(bbox, np.ndarray):
        h5_fout.create_dataset(
            'bbox', data=bbox,
            compression='gzip', compression_opts=1,
            dtype=bbox_dtype,
        )       
    h5_fout.close()
    

def load_h5_basic(filename): 
    f = h5py.File(filename, 'r')
    data = f['data'][:]
    if data.shape[0] in [3,4]: # data has beeen saved with dimension permutation
        return data.T
    return data


def load_h5(h5_filename, bbox=False):
    f = h5py.File(h5_filename, 'r')
    # f.keys() should be [u'data', u'label']
    data = f['data'][:]
    label = f['label'][:]
    if bbox: 
        bbox = f['bbox'][:]
        return (data, label, bbox)
    return (data,label)


def get_data_files(data_dir):
    """ Retrieves a list from data_files to train/test with from a txt file.

    Args:
        data_dir (string): the path to the txt file
    """
    data_files = [x.strip() for x in open(data_dir).readlines()]
    return data_files


def load_json_file(label_file):
    """ Loads json file to a python dictionary.

    Args:
        label_file: full file path to json file

    Returns:
        dictionary contained in json file 
    """
    with open(label_file) as f:
        data = json.load(f)
    return data 

