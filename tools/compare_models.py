import pickle 
import argparse
import glob
import time
from pathlib import Path
import pickle 

import matplotlib.pyplot as plt

from pcdet.config import (cfg, cfg_from_list, cfg_from_yaml_file,
                          log_config_to_file)
from pcdet.datasets.JRDB.jrdb_dataset import JrdbDataset
from pcdet.datasets.custom.eval.evaluate import * 
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.utils import plot_utils

import logging

COLORS = [
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
        '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
        '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
        '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']


class Evaluator(object): 
    def __init__(self, cfg_file, logger=False ):
        self.logger = False 
        if logger: 
            self.logger = True
            logging.basicConfig(filename="../results/log2.txt",
                            level=logging.INFO,
                            format='%(levelname)s: %(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S')

        cfg_from_yaml_file(cfg_file, cfg)
        self.model_dir = Path('../results')

        self.dataset = JrdbDataset(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=['Pedestrian'],
                root_path=None,
                training=False,
                logger=None,
            )

        self.num_frames = len(self.dataset.data_infos)
        print(self.num_frames)

        self.class_names = self.dataset.class_names
        self.easy_eval = True 

    def get_pr_values(self, exp_list=None):
        self.dataset.get_all_labels()
        print('Loaded all labels')
        flag = False 
        if not exp_list: 
            exp_list = self.model_dir.glob('*.pkl')
            flag = True
        for result in exp_list: 
            if flag:
                exp_name = result.stem
            else: 
                exp_name = result 
                filename = exp_name +'.pkl'
                result = self.model_dir / filename
            if self.logger: 
                logging.info('======= EXPERIMENT: {} ======='.format(exp_name))
            final_output_dir = self.model_dir / exp_name
            final_output_dir.mkdir(parents=True, exist_ok=True)    
            # result = '/home/crowdbot/master_lara/OpenPCDet/output/kitti_models/pointrcnn_no_pretrained/jrdb_exp24_aug_wo_gt_no_pretrain_corrected_iou/eval/epoch_30/val/jrdb/result.pkl'
            with open(result, 'rb') as f: 
                det_annos = pickle.load(f)
            if self.logger: 
                logging.info('---> Loaded {} det annos'.format(len(det_annos)))

            result_str, result_dict = self.dataset.evaluation(
                det_annos, self.class_names,
                eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
                output_path=final_output_dir, 
                easy_eval=self.easy_eval
            )
            if self.logger: 
                logging.info(result_str)
    
    def create_plot(self, result_list):

        iou_list = [0.3, 0.5]
        for iou in iou_list: 
            ax = None
            for i, exp_name in enumerate(result_list): 
                result_file = self.model_dir / exp_name / 'pr_results.pkl'
                pr_dict = pickle.load(open(result_file, 'rb'))
                cum_precs = pr_dict[iou]['prec']
                cum_recs = pr_dict[iou]['rec']
                legend = "{}".format(exp_name)
                ax = plot_pr_curve_ax(cum_precs, cum_recs, label=legend, color=COLORS[i], ax=ax)
                plt.legend(loc='lower right', title='Experiment', frameon=True, fontsize='medium')
            plt.title('Constraint Evaluation with Iou={}'.format(iou))
            filename= self.model_dir / 'pr_curves_experiments_iou_{}.pdf'.format(iou)
            plt.savefig(filename, bbox_inches="tight")


if __name__ == '__main__': 
    cfg_file = 'cfgs/kitti_models/pointrcnn.yaml'
    Evaluator = Evaluator(cfg_file, logger=True)
    # exp_list = ['jrdb_exp23_epoch30','jrdb_exp24_epoch30','jrdb_exp25_epoch30', 'jrdb_exp30_epoch30']
    exp_list = ['jrdb_exp26_epoch19', 'jrdb_exp27_epoch30']
    # Evaluator.create_plot(exp_list)
    Evaluator.get_pr_values(exp_list)