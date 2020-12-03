#!/usr/bin/env python
'''
    Script generating a text file with all files to be used for training and testing 
    from jrdb dataset.
'''
import argparse
import os
import re
from pathlib import Path

from sklearn.model_selection import train_test_split  

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--split', type=str, required=True, help='Indicate if train or test split')
parser.add_argument('--official', action='store_true', default=False, help='if True, it will take the recommended val/train split from the development kit readme file ')
parser.add_argument('--indoor_only', action='store_true', default=False, help='Only take indoor scenes of JRDB')
args = parser.parse_args()

DATA_DIR = '/hdd/master_lara_data/JRDB/cvgl/group/jrdb/data/'
_JRDB_VAL_SEQUENCES = [
    'clark-center-2019-02-28_1',
    'gates-ai-lab-2019-02-08_0',
    'huang-2-2019-01-25_0',
    'meyer-green-2019-03-16_0',
    'nvidia-aud-2019-04-18_0',
    'tressider-2019-03-16_1',
    'tressider-2019-04-26_2'
]

_JRDB_OUTDOOR_SCENES = [
    'clark-center-2019-02-28_0',
    'clark-center-2019-02-28_1',
    'clark-center-intersection-2019-02-28_0',
    'gates-to-clark-2019-02-28_1',
    'hewlett-packard-intersection-2019-01-24_0',
    'huang-lane-2019-02-12_0',
    'memorial-court-2019-03-16_0', 
    'meyer-green-2019-03-16_0', 
    'tressider-2019-03-16_0', 
    'tressider-2019-03-16_1'
]


def get_data_files(split, indoor_only=False):
    """ Walk through given directory and corresponding subdirectories to generate a text file 
        with all training/testing instances in a list, like it has been done for the KITTI dataset.

    Args:
        split (string): create file for either training or testing 

    Returns:
        data_files
    """

    folders_of_interest = [] 
    data_files = []       
                       
    path  = Path(DATA_DIR)
    for i in path.glob('**/**/*.h5'):
        if split in str(i): # test or train 
            if 'upper' in str(i): # take list from upper velodyne and fuse pointclouds with lower velodyne later 
                temp = [x.start() for x in re.finditer('/', str(i))]
                part_file_name = str(i)[temp[7]:] # get relevant chain of subdirectories

                if indoor_only: 
                    if not any(s in part_file_name for s in _JRDB_OUTDOOR_SCENES): 
                        data_files += [part_file_name]
                    else:
                        continue
                else: 
                    data_files += [part_file_name]
    return data_files


def train_val_split(data_files, official=False): 
    """ Create a train/test/val split of the data and save it to txt files. 

    Args:
        data_files [array]: array containing all path names of h5 files to include in database
        official [bool]: use offically recommended train/val split 
    """
    print('Full length of data: {}'.format(len(data_files)))
    if official: 
        data_train = []
        data_val = []
        for f in data_files:
            sequence = f.split('/')[-2]
            if sequence in _JRDB_VAL_SEQUENCES: data_val.append(f)
            else: data_train.append(f)
    else:     
        # create 70/30 train test split on randomly shuffled data instances 
        data_train, data_val = train_test_split(data_files, shuffle=True, train_size=0.7)
        
    dataset = {'train': data_train, 'val': data_val}
    return dataset


def save_to_file(split, frames):
    save_path = '../../data/jrdb/{}.txt'.format(split)
    print('Saving item to {}'.format(save_path))
    with open(save_path, 'w') as f:
        for item in frames:
            f.write("%s\n" % item)


if __name__=='__main__':

    frames = get_data_files(args.split, args.indoor_only)
    if args.split == 'train': 
        dataset = train_val_split(frames, args.official)
        save_to_file('val', dataset['val'])
        save_to_file('train', dataset['train'])
    else: 
        save_path('test', frames)


    
