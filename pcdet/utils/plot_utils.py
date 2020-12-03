import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import os

from ..utils.box_utils import boxes_to_corners_3d


def draw_bev(corners, ax, c="red"):
    # side and back boarder
    xy = corners[[0, 3, 2,1],:2]
    ax.plot(xy[:,0], xy[:,1], c=c, linestyle="-")
    # front boarder
    xy = corners[[0, 1], :2]
    ax.plot(xy[:,0], xy[:,1], c=c, linestyle="--")
    

def plot_dets_bev(pts_lidar,  gt_boxes, image=None, frame=None, show_intensity=False, save_fig=False, extra_tag='', extra_points=None):
    fig = plt.figure(figsize=(20, 15))
    if image is None: 
        gs = GridSpec(1, 2, figure=fig)
    else: 
        # image
        gs = GridSpec(2, 2, figure=fig)
        ax_im = fig.add_subplot(gs[1, :])
        ax_im.cla()
        ax_im.axis("off")
        ax_im.imshow(image)

    ax_bev = fig.add_subplot(gs[0, 0])
    ax_bev_zoom = fig.add_subplot(gs[0, 1])

    color_pool = np.random.uniform(size=(100, 3))
    corners_lidar = boxes_to_corners_3d(gt_boxes, rot_mat_alt=True)
    if type(frame) == str: 
        frame_name = '/'.join(frame.split('/')[-2:])
        save_name = extra_tag + str(frame_name.replace('/','_').replace('.h5', '.png')) 
    else: 
        frame_name = f'sample_idx_{frame}'
        save_name = extra_tag + frame_name + '.png'
    # BEV
    for ax in [ax_bev, ax_bev_zoom]: 
        ax.cla()
        # ax.set_aspect("equal")
        plt.xticks(fontsize= 16)
        plt.yticks(fontsize= 16)
        ax.set_xlabel("x [m]", fontsize=18)
        ax.set_ylabel("y [m]", fontsize=18)
        # ax.set_aspect('equal')
        if show_intensity:
            ax.scatter(pts_lidar[:,0], pts_lidar[:,1], s=1, c=pts_lidar[:,3], cmap='viridis')
        else: 
            ax.scatter(pts_lidar[:,0], pts_lidar[:,1], s=1, c='b')
        for i in range(corners_lidar.shape[0]):
            corners_lidar2d = corners_lidar[i, :4, :2]
            draw_bev(corners_lidar2d, ax, c=color_pool[i])
        if extra_points is not None: 
            ax.scatter(extra_points[:,0], extra_points[:,1], s=1, c='r')

    ax_bev.set_xlim(-15,15)
    ax_bev.set_ylim(-20,20)
    ax_bev_zoom.set_xlim(-5,10)
    ax_bev_zoom.set_ylim(-5,5)

    ax_bev.set_title(f"Frame: {frame_name}", fontsize=18)
    ax_bev_zoom.set_title(f"Frame zoomed: {frame_name}", fontsize=18)

    if save_fig:  
        plot_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(plot_dir, "../../plots/", save_name) 
        print(f"Saving to {plot_dir}")
        plt.savefig(plot_dir, bbox_inches='tight')  
    else: 
        print("Did not save figure. Add filename in arguments to be able to save figure.")
    
    plt.cla()
    plt.clf()
    plt.close('all')


def plot_dets_gt_bev(pts_lidar,  gt_boxes, det_boxes, frame, image=None, show_intensity=False, save_fig=False, extra_tag='', extra_points=None):
    fig = plt.figure(figsize=(20, 15))
    if image is None: 
        gs = GridSpec(1, 2, figure=fig)
    else: 
        # image
        gs = GridSpec(2, 2, figure=fig)
        ax_im = fig.add_subplot(gs[1, :])
        ax_im.cla()
        ax_im.axis("off")
        ax_im.imshow(image)

    ax_bev_gt = fig.add_subplot(gs[0, 0])
    ax_bev_det = fig.add_subplot(gs[0, 1])

    color_pool = np.random.uniform(size=(100, 3))
    corners_lidar_gt = boxes_to_corners_3d(gt_boxes, rot_mat_alt=True)
    corners_lidar_det = boxes_to_corners_3d(det_boxes, rot_mat_alt=True)

    if type(frame) == str: 
        frame_name = '/'.join(frame.split('/')[-2:])
        save_name = extra_tag + str(frame_name.replace('/','_').replace('.h5', '.png')) 
    else: 
        frame_name = f'sample_idx_{frame}'
        save_name = extra_tag + frame_name + '.png'
    # BEV
    for ax in [ax_bev_gt, ax_bev_det]: 
        ax.cla()
        # ax.set_aspect("equal")
        plt.xticks(fontsize= 16)
        plt.yticks(fontsize= 16)
        ax.set_xlabel("x [m]", fontsize=18)
        ax.set_ylabel("y [m]", fontsize=18)
        ax.set_xlim(-5,10)
        ax.set_ylim(-10,10)
    if show_intensity:
        ax_bev_gt.scatter(pts_lidar[:,0], pts_lidar[:,1], s=1, c=pts_lidar[:,3], cmap='viridis')
    else: 
        ax_bev_gt.scatter(pts_lidar[:,0], pts_lidar[:,1], s=1, c='b')
    
    if extra_points is not None: 
        ax_bev_det.scatter(extra_points[:,0], extra_points[:,1], s=1, c='r')

    for i in range(corners_lidar_gt.shape[0]):
        corners_lidar2d = corners_lidar_gt[i, :4, :2]
        draw_bev(corners_lidar2d, ax_bev_gt, c=color_pool[i])
    
    for i in range(corners_lidar_det.shape[0]):
        corners_lidar2d = corners_lidar_det[i, :4, :2]
        draw_bev(corners_lidar2d, ax_bev_det, c=color_pool[i])

    ax_bev_gt.set_title(f"Frame: {frame_name} - ground truth", fontsize=18)
    ax_bev_det.set_title(f"Frame: {frame_name} - after sampling/augmentation", fontsize=18)

    if save_fig:  
        plot_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(plot_dir, "../../plots/", save_name) 
        plt.savefig(plot_dir, bbox_inches='tight')  
        print(f"Saving to {plot_dir}")
    else: 
        print("Did not save figure. Add filename in arguments to be able to save figure.")

    plt.cla()
    plt.clf()
    plt.close('all')
