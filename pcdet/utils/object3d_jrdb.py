import numpy as np

def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

class Object3d(object):
    def __init__(self, line, gt=False): # if read from ground truth label, the txt file looks different 
        if gt: # read from label file (annotation in velodyne coord sys)
            bbox = line 
            # line is in this case the bbox itself 
            self.loc = np.array((float(bbox[0]), float(bbox[1]), float(bbox[2])), dtype=np.float32) # in velo coords
            self.l, self.w, self.h = float(bbox[3]), float(bbox[4]), float(bbox[5])
            self.ry =  float(bbox[6])  # rotation angle around z-axis (instead of y as in camera coord.)
            # self.dis_to_cam = np.linalg.norm(self.loc)
            # According to KITTI definition
            self.cls_type = 'Pedestrian'
            self.cls_id = 2
            beta = np.arctan2(self.loc[1], self.loc[0])
            self.alpha = -np.sign(beta) * np.pi / 2 + beta + self.ry
            self.score = -1.0
        
        else: # read from detection file including more information 
            label = line.strip().split(' ')
            self.src = line
            self.cls_type = label[0]
            self.cls_id = cls_type_to_id(self.cls_type)
            self.alpha = float(label[1])
            # self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
            self.h = float(label[5])
            self.w = float(label[6])
            self.l = float(label[7])
            self.loc = np.array((float(label[2]), float(label[3]), float(label[4])), dtype=np.float32)
            # self.dis_to_cam = np.linalg.norm(self.loc)
            self.ry = float(label[8])
            self.score = float(label[9]) if label.__len__() == 10 else -1.0


    def to_str(self):
        print_str = '%s %.3f pos: %s hwl: [%.3f %.3f %.3f] ry: %.3f' \
                     % (self.cls_type, self.alpha, self.pos, self.h, self.w, self.l,
                        self.ry)
        return print_str

    def to_det_format(self):
        print_str = '%s %.3f %s %.3f %.3f %.3f %.3f %.3f %.3f' \
                     % (self.cls_type, self.alpha, self.pos[0],self.pos[1], self.pos[2], self.h, self.w, self.l, self.ry)
        return print_str



