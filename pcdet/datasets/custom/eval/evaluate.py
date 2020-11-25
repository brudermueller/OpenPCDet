'''
    Evaluation code for custom dataset in order to compute average precision etc.
    Code adapted from https://github.com/Kartik17/PointRCNN-Argoverse. 
'''


import argparse
import collections
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from numba import jit
from scipy.spatial.transform import Rotation

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils, custom_data_utils
from ...kitti.kitti_object_eval_python.eval import calculate_iou_partly

COLORS = [
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
        '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
        '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
        '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']


def get_AP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100

def save_fig(fig, path, extension='.pdf'):
    save_dir = path + extension
    fig.savefig(save_dir, bbox_inches='tight')


def plot_pr_curve_ax(
    precisions, recalls, label=None, color=None, ax=None):
    """Simple plotting helper function"""

    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.gca()

    if color is None:
        color = COLORS[0]
    ax.scatter(recalls, precisions, label=label, s=20, color=color)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    title= 'Precision-Recall curve for human class'
    ax.set_title(title)
    ax.set_xlim([0.0,1.1])
    ax.set_ylim([0.0,1.1])
    return ax

def compute_ap(tp_list, num_gt, num_pred):

    assert num_gt != 0
    assert num_pred != 0
    tp = tp_list.sum()
    # Compute precision and recall at each prediction box step
    cum_precisions = np.cumsum(tp_list) / (np.arange(num_pred) + 1)
    cum_recalls = np.cumsum(tp_list).astype(np.float32) / num_gt
    
    # Pad with start and end values to simplify the math
    cum_precisions = np.concatenate([[0], cum_precisions, [0]])
    cum_recalls = np.concatenate([[0], cum_recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(cum_precisions) - 2, -1, -1):
        cum_precisions[i] = np.maximum(cum_precisions[i], cum_precisions[i + 1])

    # Compute AP 
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    indices = np.where(cum_recalls[:-1] != cum_recalls[1:])[0] + 1
    AP = np.sum((cum_recalls[indices] - cum_recalls[indices - 1]) *
                 cum_precisions[indices])
    precision = tp / num_pred
    recall = tp / num_gt
    return AP, cum_precisions, cum_recalls, precision, recall


def get_tp_mask_from_overlaps(pred_boxes, gt_boxes, iou_thresholds): 
    pred_boxes = torch.from_numpy(pred_boxes).contiguous().cuda(non_blocking=True).float()
    gt_boxes = torch.from_numpy(gt_boxes).contiguous().cuda(non_blocking=True).float()
    iou3d = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes, gt_boxes)  # (M, N)
    max_iou, gt_ind  = torch.max(iou3d, dim=1) # max_overlaps, gt_assignment

    mask_dict = {}
    for iou in iou_thresholds: 
        assigned_gt_box_idx = []
        true_positive_mask = []
        for el in zip(max_iou,gt_ind): 
            if el[0].item() > iou and el[1].item() not in assigned_gt_box_idx: 
                true_positive_mask.append(1.0)
            else: 
                true_positive_mask.append(0.0)
            
        true_positive_mask = (max_iou > iou).float().cpu().numpy()
        mask_dict[iou] = true_positive_mask
    return mask_dict
        
def get_results_distributed(pred_boxes, gt_boxes, scores_np, tp_mask_dict, save_path): 
    """
    Calculate results after having performed iou overlap calculation per 
    frame separately. 

    Args:
        pred_boxes (numpy ndarrays)
        gt_boxes (numpy ndarrays)
        scores (numpy ndarrays)
    """
    ax = None
    AP_dict = {}
    for i, (iou, tp_mask) in enumerate(tp_mask_dict.items()): 
        no_dets_ct = 0 
        iou = np.round(iou, decimals=2) 
        preds = np.empty([0,10])
        num_preds = pred_boxes.shape[0]
        num_gt_boxes = gt_boxes.shape[0]

        preds=np.hstack((pred_boxes, 
                         scores_np.reshape(num_preds,-1), 
                         tp_mask.reshape(num_preds,-1))) 
        # sort all detections by score 
        preds_final = preds[np.argsort(preds[:, 7])[::-1]]
        AP, cum_precisions, cum_recalls, precision, recall = compute_ap(preds_final[:, 8], num_gt_boxes, num_preds)
        AP_dict['AP@iou:{}'.format(iou)] = AP 
        # print('Precision: {}\nRecall: {}'.format(precision, recall))
        # print('Average Precision: {}'.format(AP))
        legend = "IOU={}: AP={:.1%}, Recall={:05.4f}".format(iou, AP, recall)
        ax = plot_pr_curve_ax(cum_precisions, cum_recalls, label=legend, color=COLORS[i], ax=ax)

    plt.legend(loc='lower right', title='IOU Thresh', frameon=True, fontsize='medium')

    for xval in np.linspace(0.0, 1.0, 11):
        plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')

    #     fig_name = "PRCurve_val_" + config['name']
    save_dir = save_path / '../plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    filename= save_dir / 'pr_curve.pdf'
    plt.savefig(filename, bbox_inches="tight")
    plt.show()

    result_str = '\n'
    for iou, AP in AP_dict.items(): 
        result_str += 'AP@IoU{}: {}\n'.format(iou, AP) 
    return result_str, AP_dict


def get_results(pred_boxes_np, gt_boxes_np, scores_np, save_path): 
    """ 
    Get results from all predicions and ground truth boxes over all frames. 
    Warning: this function might lead to memory issues in GPU since iou3d 
    overlap calculation is performed on the GPU. 

    Args:
        pred_boxes (numpy ndarrays)
        gt_boxes (numpy ndarrays)
        scores (numpy ndarrays)
    """
    ax = None
    # frame_no_dets = collections.defaultdict(list)	
    iou_thresholds = np.arange(0.3, 0.9, 0.05)
    pred_boxes = torch.from_numpy(pred_boxes_np).contiguous().cuda(non_blocking=True).float()
    gt_boxes = torch.from_numpy(gt_boxes_np).contiguous().cuda(non_blocking=True).float()
    scores = torch.from_numpy(scores_np).contiguous().cuda(non_blocking=True).float()

    AP_dict = {}
    for i, iou in enumerate(iou_thresholds): 
        no_dets_ct = 0 
        iou = np.round(iou, decimals=2) 
        # print('IoU = {}'.format(iou))
        preds = np.empty([0,10])

        norm_scores = torch.sigmoid(scores)
        scores_sorted,_ = torch.sort(norm_scores, dim=0, descending=True) # sort scores 
        # print(scores_sorted.min().item(), scores_sorted.max().item())
        scores_idx = torch.argsort(norm_scores, dim=0, descending=True)  # keep track of original indices 
        # filter_ids = torch.where(scores_sorted > score_thresh) # Remove boxes with scores lower than threshold scores

        # scores_idx = scores_idx[filter_ids]
        # pred_boxes = pred_boxes[scores_idx] # order predictions with descending score 
        num_preds = pred_boxes.shape[0]
        num_gt_boxes = gt_boxes_np.shape[0]

        if num_preds == 0: 
            # frame_no_dets["{}: {}".format(idx,frame)].append(iou)
            no_dets_ct +=1
            continue 
        
        iou3d = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes, gt_boxes)  # (M, N)
        max_iou, gt_ind  = torch.max(iou3d, dim=1) # max_overlaps, gt_assignment
        
        assigned_gt_box_idx = []
        true_positive_mask = []
        for el in zip(max_iou,gt_ind): 
            if el[0].item() > iou and el[1].item() not in assigned_gt_box_idx: 
                true_positive_mask.append(1.0)
            else: 
                true_positive_mask.append(0.0)
            
        true_positive_mask = (max_iou > iou).float().cpu().numpy()

        # convert to numpy arrays 
        pred_boxes_np = pred_boxes.cpu().numpy()
        # scores_np = scores[scores_idx].cpu().numpy()
        scores_np = scores.cpu().numpy()
        preds=np.hstack((pred_boxes_np, 
                         scores_np.reshape(num_preds,-1), 
                         true_positive_mask.reshape(num_preds,-1), 
                         gt_ind.cpu().numpy().reshape(num_preds,-1)))
        
        # sort all detections by score 
        preds_final = preds[np.argsort(preds[:, 7])[::-1]]
        AP, cum_precisions, cum_recalls, precision, recall = compute_ap(preds_final[:, 8], num_gt_boxes, num_preds)
        AP_dict['AP@iou:{}'.format(iou)] = AP 
        # print('Precision: {}\nRecall: {}'.format(precision, recall))
        # print('Average Precision: {}'.format(AP))
        legend = "IOU={}: AP={:.1%}, Recall={:05.4f}".format(iou, AP, recall)
        ax = plot_pr_curve_ax(cum_precisions, cum_recalls, label=legend, color=COLORS[i], ax=ax)

    plt.legend(loc='lower right', title='IOU Thresh', frameon=True, fontsize='medium')

    for xval in np.linspace(0.0, 1.0, 11):
        plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')

    #     fig_name = "PRCurve_val_" + config['name']
    save_dir = save_path / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    filename= save_dir / 'pr_curve_test.pdf'
    plt.savefig(filename, bbox_inches="tight")
    plt.show()
    print('Frames without detections: {}'.format(no_dets_ct))

    result_str = '\n'
    for iou, AP in AP_dict.items(): 
        result_str += 'AP@IoU{}: {}\n'.format(iou, AP) 
    return result_str, AP_dict
