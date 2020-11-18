import os
import numpy as np 

from pcdet.utils.custom_data_utils import load_h5, get_data_files

def main(): 
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/custom_data/")
    all_files = get_data_files(os.path.join(DATA_DIR, 'train.txt'))


    for frame in all_files: 
        lidar_file = os.path.join(DATA_DIR, frame)
        assert os.path.exists(lidar_file)
        pts, _ = load_h5(lidar_file)
        print(pts.shape)
        path = os.path.join(DATA_DIR, '{}.npy'.format(frame))
        np.save(path, pts)

if __name__ == "__main__":
    main()    
