'''
    Evaluation code for custom dataset in order to compute average precision etc.
    Code adapted from https://github.com/Kartik17/PointRCNN-Argoverse. 
'''


import argparse
import collections
import json
import os
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np
import numba 
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


def compute_ap_aos(tp_list, ang_sim_list, num_gt, num_pred, eleven_pt_interolation=True):

    assert num_gt != 0
    assert num_pred != 0
    tp = tp_list.sum()
    # Compute precision and recall at each prediction box step
    cum_precisions = np.cumsum(tp_list) / (np.arange(num_pred) + 1)
    cum_recalls = np.cumsum(tp_list).astype(np.float32) / num_gt
    cum_os = np.cumsum(ang_sim_list) / (np.arange(num_pred) + 1)
    
    if eleven_pt_interolation: 
        prec_at_rec = []
        orient_sim_at_rec = []
        for recall_level in np.linspace(0.0, 1.0, 11):
            try:
                args = np.argwhere(cum_recalls >= recall_level).flatten()
                prec = np.max(cum_precisions[args])
                orient_sim = np.max(cum_os[args])
            except ValueError:
                prec = 0.0
                orient_sim = 0.0
            prec_at_rec.append(prec)
            orient_sim_at_rec.append(orient_sim)
        AP = np.mean(prec_at_rec)
        AHS = np.mean(orient_sim_at_rec)
    else: # interpolating all points
        # Pad with start and end values to simplify the math
        cum_precisions = np.concatenate([[0], cum_precisions, [0]])
        cum_recalls = np.concatenate([[0], cum_recalls, [1]])
        # Ensure precision values decrease but don't increase. This way, the
        # precision value at each recall threshold is the maximum it can be
        # for all following recall thresholds, as specified by the VOC paper.
        for i in range(len(cum_precisions) - 2, -1, -1):
            cum_precisions[i] = np.maximum(cum_precisions[i], cum_precisions[i + 1])
            cum_os[i] = np.maximum(cum_os[i], cum_os[i + 1])
        # Compute AP 
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        indices = np.where(cum_recalls[:-1] != cum_recalls[1:])[0] + 1
        AP = np.sum((cum_recalls[indices] - cum_recalls[indices - 1]) *
                    cum_precisions[indices])
        AHS = np.sum((cum_recalls[indices] - cum_recalls[indices - 1]) *
                    cum_os[indices])
                                 
    precision = tp / num_pred
    recall = tp / num_gt
    return AP, AHS, cum_precisions, cum_recalls, precision, recall


# @numba.jit
def get_tp_mask_from_overlaps(pred_boxes, gt_boxes, iou_thresholds): 
    pred_boxes = torch.from_numpy(pred_boxes).contiguous().cuda(non_blocking=True).float()
    gt_boxes = torch.from_numpy(gt_boxes).contiguous().cuda(non_blocking=True).float()
    # calculate overlaps based on 3D iou
    iou3d = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes, gt_boxes)  # (M, N)
    max_iou, gt_ind  = torch.max(iou3d, dim=1) # max_overlaps, gt_assignment

    tp_mask_dict = {}

    for iou in iou_thresholds: 
        true_positive_mask = 0.0 *  np.ones([pred_boxes.shape[0]])
        assigned_box_dct = {} # mapping temporary chosen prediction idx to box idx 

        # iterate through list of ground truth boxes 
        for j in range(gt_boxes.shape[0]):
            max_overlap = -100
            # iterate through list of detections (M)
            for i in range(pred_boxes.shape[0]): 
                overlap_val = max_iou[i].item()
                gt_idx = gt_ind[i].item()
                if j == gt_idx and overlap_val > iou and overlap_val > max_overlap: 
                    max_overlap = max_iou[i].item()
                    true_positive_mask[i] = 1.0
                    old_idx = assigned_box_dct.get(j, None)
                    if old_idx: 
                        true_positive_mask[old_idx] = 0.0
                    assigned_box_dct[j] = i 

        tp_mask_dict[iou] = true_positive_mask
    return tp_mask_dict, gt_ind.cpu().numpy()

# @numba.jit     
def get_results_distributed(pred_boxes, gt_boxes, scores_np, tp_mask_dict, angular_similarity_dict, save_path): 
    """
    Calculate results after having performed iou overlap calculation per 
    frame separately. 

    Args:
        pred_boxes (numpy ndarrays)
        gt_boxes (numpy ndarrays)
        scores (numpy ndarrays)
    """
    ax = None
    result_dict = {}
    precision_recall_dict = {}
    result_str = '\n'
    for i, (iou, tp_mask) in enumerate(tp_mask_dict.items()): 
        ang_sim_list = angular_similarity_dict[iou]
        no_dets_ct = 0 
        iou = np.round(iou, decimals=2) 
        preds = np.empty([0,10])
        num_preds = pred_boxes.shape[0]
        num_gt_boxes = gt_boxes.shape[0]

        preds=np.hstack((pred_boxes, 
                         scores_np.reshape(num_preds,-1), 
                         ang_sim_list.reshape(num_preds, -1),
                         tp_mask.reshape(num_preds,-1))) 
        # sort all detections by score 
        sorted_scores_idx = np.argsort(scores_np)[::-1]
        preds_final = preds[sorted_scores_idx]

        AP, AHS, cum_precisions, cum_recalls, precision, recall = compute_ap_aos(
            preds_final[:, -1], preds_final[:,-2],num_gt_boxes, num_preds)

        result_str += 'AP@IoU{}: {:.3f}\n'.format(iou, AP) 
        result_str += 'AHS@IoU{}: {:.3f}\n\n'.format(iou, AHS) 

        result_dict['AP@iou_{}'.format(iou)] = AP 
        result_dict['AHS@iou_{}'.format(iou)] = AHS

        # print('Precision: {}\nRecall: {}'.format(precision, recall))
        # print('Average Precision: {}'.format(AP))
        legend = "IOU={}: AP={:.1%}, Recall={:05.4f}".format(iou, AP, recall)

        precision_recall_dict[iou] = {'prec': cum_precisions, 'rec': cum_recalls}
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
    save_file = save_path / 'pr_results.pkl'
    
    with open(save_file, 'wb') as f:
        pickle.dump(precision_recall_dict, f)

    return result_str, result_dict


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



@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps,
                           gt_datas,
                           dt_datas,
                           dt_scores,
                           ignored_gt,
                           ignored_det,
                           metric,
                           min_overlap,
                           thresh=0,
                           compute_fp=False,
                           compute_aos=False):

    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True

    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0

    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap)
                    and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score
            elif (compute_fp and (overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif (compute_fp and (overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1
        
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
       
        elif valid_detection != NO_DETECTION:
            tp += 1
            # thresholds.append(dt_scores[det_idx])
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                # delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta[delta_idx]  = gt_datas[i, 6] - gt_boxes[det_idx,6]
                delta_idx += 1
            assigned_detection[det_idx] = True
    
    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1
                     or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1

        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            # tmp = [0] * fp
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0

            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             dc_nums,
                             gt_datas,
                             dt_datas,
                             ignored_gts,
                             ignored_dets,
                             metric,
                             min_overlap,
                             thresholds,
                             compute_aos=False):
    gt_num = 0
    dt_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
                               gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_det,
                metric,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]


def get_results_jit(gt_boxes, dt_boxes, scores, ignored_gts, ignored_dets, min_overlaps, save_path):
    for  k, min_overlap in enumerate(min_overlaps):
        pred_boxes = torch.from_numpy(dt_datas).contiguous().cuda(non_blocking=True).float()
        gt_boxes = torch.from_numpy(gt_datas).contiguous().cuda(non_blocking=True).float()
        # calculate overlaps based on 3D iou
        overlaps = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes, gt_boxes).cpu().numpy()  # (M, N)
        # max_iou, gt_ind  = torch.max(iou3d, dim=1) # max_overlaps, gt_assignment
        thresholdss = []
        for i in range(len(gt_annos)):
            rets = compute_statistics_jit(
                overlaps, 
                gt_boxes[i],
                dt_boxes[i],
                ignored_gts[i],
                ignored_dets[i],
                dontcares[i],
                min_overlap=min_overlap,
                thresh=0.0,
                compute_fp=True)
            tp, fp, fn, similarity, thresholds = rets
            thresholdss += thresholds.tolist()
        thresholdss = np.array(thresholdss)
        thresholds = get_thresholds(thresholdss, total_num_valid_gt)
        thresholds = np.array(thresholds)
        pr = np.zeros([len(thresholds), 4]) 
        fused_compute_statistics(
                        parted_overlaps[j],
                        pr,
                        total_gt_num[idx:idx + num_part],
                        total_dt_num[idx:idx + num_part],
                        total_dc_num[idx:idx + num_part],
                        gt_datas_part,
                        dt_datas_part,
                        dc_datas_part,
                        ignored_gts_part,
                        ignored_dets_part,
                        metric,
                        min_overlap=min_overlap,
                        thresholds=thresholds,
                        compute_aos=compute_aos)
        for i in range(len(thresholds)):
            recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
            precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
            if compute_aos:
                aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
        for i in range(len(thresholds)):
            precision[m, l, k, i] = np.max(
                precision[m, l, k, i:], axis=-1)
            recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
            if compute_aos:
                aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
    ret_dict = {
        "recall": recall,
        "precision": precision,
        "orientation": aos,
    }
