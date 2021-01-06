#!/usr/bin/env python
'''
    Script generating a text file with all files to be used for training and testing 
    from custom dataset.
'''
import os 
import argparse
import re 
from pathlib import Path

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--datadir', type=str, default='custom_data', required=True, help='specify the dataset to use')
parser.add_argument('--file_name', type=str, default='full_data.txt', help='Indicate filename to save file list to')
parser.add_argument('--savedir', type=str, default='data/custom_data', required=False, help='specify the savedir to use')
args = parser.parse_args()

DATA_DIR = Path(args.datadir)


def get_crowd_files(path): 
    # bag_list = list(DATA_DIR.glob('*/*.h5'))  
    bag_list = list(DATA_DIR.glob('*.h5'))  

    print(len(bag_list))
    crowd_files = [str(f) for f in bag_list] # if 'crowd' in str(f).split('/')[-2]]  
    final_out_list = []
    for f in crowd_files: 
        temp = [x.start() for x in re.finditer('/', str(f))]
        part_file_name = str(f)[temp[1]+1:] # get relevant chain of subdirectories
        final_out_list.append(part_file_name)
    return final_out_list


if __name__=='__main__':
    if args.datadir == 'LCAS': 
        path = os.path.join(DATA_DIR, 'labels')
    else: 
        path = DATA_DIR  
    # files = get_data_files(path)
    files = get_crowd_files(path)

    if args.savedir: 
        save_path = os.path.join(args.savedir, args.file_name) 
    elif 'test' in args.datadir:
        # go one directory up if we are in testing folder
        save_path = os.path.join(DATA_DIR, '..', args.file_name)
    else: 
        save_path = os.path.join(str(DATA_DIR.resolve()), args.file_name)
    print(args.datadir)
    print('Saving item to {}'.format(save_path))
    with open(save_path, 'w') as f:
        for item in sorted(files, key=lambda x: int(re.findall(r'\d+', x.split('.')[0].split('/')[-1])[0])):
        # for item in files: 
            f.write("%s\n" % item)