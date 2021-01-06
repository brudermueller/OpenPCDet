import numpy as np 
from mayavi import mlab

def draw_lidar_cam(pc, color=None, fig=None):
    ''' Simple setup to draw lidar points in camera coordinate system.'''
    
    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:,1]
    #draw points
    mlab.points3d(pc[:,2], -pc[:,0], -pc[:,1], color, color=None, mode='point', colormap = 'gnuplot', scale_factor=1, figure=fig)
    #draw origin
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
    #draw axis
    axes=np.array([
        [1.,0.,0.,0.],
        [0.,1.,0.,0.],
        [0.,0.,1.,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=30.0, figure=fig)
    return fig

def draw_lidar(pc, color=None, fig=None, bgcolor=(0,0,0), pts_scale=1, pts_mode='point', pts_color=None):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    '''
    if fig is None: fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:,2]
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color, color=pts_color, mode=pts_mode, colormap = 'gnuplot', scale_factor=pts_scale, figure=fig)
    
    #draw origin
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
    
    #draw axis
    axes=np.array([
        [2.,0.,0.,0.],
        [0.,2.,0.,0.],
        [0.,0.,2.,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)

    # draw fov (todo: update to real sensor spec.)
    fov=np.array([  # 45 degree
        [20., 20., 0.,0.],
        [20.,-20., 0.,0.],
    ],dtype=np.float64)
    
    mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
    mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
   
    # draw square region
    TOP_Y_MIN=-20
    TOP_Y_MAX=20
    TOP_X_MIN=0
    TOP_X_MAX=40
    TOP_Z_MIN=-2.0
    TOP_Z_MAX=0.4
    
    x1 = TOP_X_MIN
    x2 = TOP_X_MAX
    y1 = TOP_Y_MIN
    y2 = TOP_Y_MAX
    mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    
    #mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig

def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=1, draw_text=True, text_scale=(0.25,0.25,0.25), color_list=None, sample_id=None):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    ''' 
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n] 
        if draw_text: 
            if not sample_id: 
                mlab.text3d(b[4,2], -b[4,0], -b[4,1], '%d'%n, scale=text_scale, color=color, figure=fig)
            else: 
                mlab.text3d(b[4,2], -b[4,0], -b[4,1], '%d'%sample_id, scale=text_scale, color=color, figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,2], b[j,2]], [-b[i,0], -b[j,0]], [-b[i,1], -b[j,1]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,2], b[j,2]], [-b[i,0], -b[j,0]], [-b[i,1], -b[j,1]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,2], b[j,2]], [-b[i,0], -b[j,0]], [-b[i,1], -b[j,1]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    #mlab.show(1)
    #mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig

def draw_sphere_pts(pts, color=(0, 1, 0), fig=None, bgcolor=(0, 0, 0), scale_factor=0.2):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()

    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(600, 600))

    if isinstance(color, np.ndarray) and color.shape[0] == 1:
        color = color[0]
        color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    if isinstance(color, np.ndarray):
        pts_color = np.zeros((pts.__len__(), 4), dtype=np.uint8)
        pts_color[:, 0:3] = color
        pts_color[:, 3] = 255
        G = mlab.points3d(pts[:, 2], -pts[:, 0], -pts[:, 1], np.arange(0, pts_color.__len__()), mode='sphere',
                          scale_factor=scale_factor, figure=fig)
        G.glyph.color_mode = 'color_by_scalar'
        G.glyph.scale_mode = 'scale_by_vector'
        G.module_manager.scalar_lut_manager.lut.table = pts_color
    else:
        mlab.points3d(pts[:, 2], -pts[:, 0], -pts[:, 1], mode='sphere', color=color,
                      colormap='gnuplot', scale_factor=scale_factor, figure=fig)

    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
    mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), line_width=3, tube_radius=None, figure=fig)

    return fig


def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig


def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)
    return fig

def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def compute_box_3d(obj, gt=False):
    ''' Computes 8 corners of bounding boxes in the camera coordinate system. 
        Returns:
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)    

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h
    
    # 3d bounding box corners
    x_corners = [w/2,w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2]
    y_corners = [-h/2.,-h/2.,-h/2.,-h/2.,h/2.,h/2.,h/2.,h/2.]
    z_corners = [l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2,l/2]
    
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners])).T
    corners_3d += obj.loc 
    return corners_3d

def compute_box_3d_velo(obj): 
    """ Computes 8 corners of bounding boxes in the velodyne coordinate system. 
    Args:
        obj: object of box class including relevant box parameters 

    Returns:
        corners3d: (8,3) numpy ndarray including all corners of the box 
    """
    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h
    # careful: width, length and height have been differently defined than in KITTI
    x_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]        
    y_corners = [-l/2, l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2]
    z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]

    # rotation now defined in Velodyne coords. -> around z-axis => yaw rot. 
    R = np.array([[np.cos(obj.ry), np.sin(obj.ry), 0],
                  [-np.sin(obj.ry), np.cos(obj.ry), 0],
                  [0, 0, 1]])
    corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
    # transpose and rotate around orientation angle 
    corners3d = np.dot(R, corners3d).T
    corners3d = corners3d + obj.loc
    return corners3d

def show_lidar_with_boxes(pc_cam, objects, gt_objects=None, best_prop_only=False,foreground=None): 
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in cam coord system) 
    '''
    
    print(('All point num: ', pc_cam.shape[0]))
    fig = mlab.figure(figure=None, bgcolor=(0,0,0),fgcolor=None, engine=None, size=(1000, 500))
    
    fig = draw_lidar_cam(pc_cam, fig=fig)
    print('Got {} detections'.format(len(objects)))
    scores = []
    if objects: 
        for idx, obj in enumerate(objects):
            scores.append(obj.score)
            # Draw 3d bounding box
            if not best_prop_only: 
                box3d_pts_3d = compute_box_3d(obj) 
                draw_gt_boxes3d([box3d_pts_3d], fig=fig)
        if best_prop_only: 
            print(np.max(scores))
            best_id = np.argmax(scores)
            best_proposal = objects[best_id]
            box3d_pts_3d = compute_box_3d(best_proposal) 
            obj = best_proposal
            mlab.points3d(obj.loc[2], -obj.loc[0], -obj.loc[1], color=(1,1,1), mode='sphere', scale_factor=0.2, figure=fig)
            
            draw_gt_boxes3d([box3d_pts_3d], fig=fig)
    
    if gt_objects is not None: 
        print('Got {} ground truth objects'.format(len(gt_objects)))
        for idx, obj in enumerate(gt_objects): 
            if type(obj) == tuple: 
                pred_box3d_pts_3d = compute_box_3d(obj[1], gt=True) 
                sample_id = obj[0]
                draw_gt_boxes3d([pred_box3d_pts_3d], fig=fig, color=(0,1,0), sample_id=sample_id)
            else: 
                # Draw 3d bounding box
                pred_box3d_pts_3d = compute_box_3d(obj, gt=True) 
                draw_gt_boxes3d([pred_box3d_pts_3d], fig=fig, color=(0,1,0))
                mlab.points3d(obj.loc[2], -obj.loc[0], -obj.loc[1], color=(1,1,1), mode='sphere', scale_factor=0.2, figure=fig)

    if foreground is not None: 
        # plot foreground segmentation results
        draw_sphere_pts(foreground, fig=fig)

    mlab.show(1)
    return fig 

def draw_bev(corners, ax, c="red"):
    # side and back boarder
    xy = corners[[3,2,1,0],:2]
    ax.plot(xy[:,0], xy[:,1], c=c, linestyle="-")
    # front boarder
    xy = corners[[0, 3], :2]
    ax.plot(xy[:,0], xy[:,1], c=c, linestyle="--")

def _plot_sequence(pts_lidar,  gt_boxes, frame, det_boxes=None, 
                   show_intensity=False, save_name=None, auto_save=False, 
                   ax_limits=None):
    ''' Plot sequence with detections and ground truth in bird's eye-view. 
        Creates two plots next to each other with varying zoom levels into point cloud. 
    Args: 
        pts_lidar [numpy ndarray]: 3d/4d depending on show_intensity 
        gt_boxes [numpy ndarray]: nx7 for (x,y,z, w,l,h,rz)
        frame: either filename (str) or frame idx (int)
        det_boxes: [numpy ndarray]: nx7 for (x,y,z, w,l,h,rz)    
        save_name [str]: filename to save plot under 
        auto_save [bool]: if true, saved automatically 
        ax_limits [numpy ndarray]: (4, 2) 
    '''
    plt.rc('xtick',labelsize=14)
    plt.rc('ytick',labelsize=14)
    fig = plt.figure(figsize=(20, 15))
    
    gs = GridSpec(1, 2, figure=fig)
    ax_bev = fig.add_subplot(gs[0, 0])
    ax_bev_zoom = fig.add_subplot(gs[0, 1])
    
    color_pool = np.random.uniform(size=(100, 3))
    corners_lidar = boxes_to_corners_3d(gt_boxes, rot_mat_alt=False)
    
    if det_boxes is not None: 
        corners_lidar_det = boxes_to_corners_3d(det_boxes, rot_mat_alt=True)
        
    if type(frame) == str: 
        frame_name = '/'.join(frame.split('/')[-2:])
        auto_save_name = str(frame_name.replace('/','_').replace('.h5', '.png')) 
    else: 
        frame_name = f'sample_idx_{frame}'
        auto_save_name = frame_name + '.png'
    
    # BEV
    for ax in [ax_bev, ax_bev_zoom]: 
        ax.cla()
        ax.set_xlabel("x [m]", fontsize=18)
        ax.set_ylabel("y [m]", fontsize=18)
        if show_intensity:
            ax.scatter(pts_lidar[:,0], pts_lidar[:,1], s=1, c=pts_lidar[:,3], cmap='viridis')
        else: 
            ax.scatter(pts_lidar[:,0], pts_lidar[:,1], s=1, c='b')
        for i in range(corners_lidar.shape[0]):
            corners_lidar2d = corners_lidar[i, :4, :2]
            draw_bev(corners_lidar2d, ax, c='r') # c=color_pool[i]
        if det_boxes is not None: 
            for i in range(corners_lidar_det.shape[0]):
                corners_lidar2d_det = corners_lidar_det[i, :4, :2]
                draw_bev(corners_lidar2d_det, ax, c='g')
    
    ax_bev.set_title(f"Frame: {frame_name}", fontsize=18)
    ax_bev_zoom.set_title(f"Frame zoomed: {frame_name}", fontsize=18)
    
    if ax_limits is not None: 
        ax_bev.set_xlim(*ax_limits[0])
        ax_bev.set_ylim(*ax_limits[1])
        ax_bev_zoom.set_xlim(*ax_limits[2])
        ax_bev_zoom.set_ylim(*ax_limits[3])
    else: 
        ax_bev.set_xlim(-5,10)
        ax_bev.set_ylim(-10,10)
        ax_bev_zoom.set_xlim(-5,5)
        ax_bev_zoom.set_ylim(-5,5)
    
    plot_dir = os.path.dirname(os.path.abspath(os.getcwd()))
    if save_name:  
        plot_dir = os.path.join(plot_dir, "../plots/results/", save_name+'_bev.pdf')
        print(f"Saving to {plot_dir}")
        plt.savefig(plot_dir, bbox_inches='tight')  
    elif auto_save: 
        plot_dir = os.path.join(plot_dir, "../plots/", auto_save_name)
        print(f"Saving to {plot_dir}")
        plt.savefig(plot_dir, bbox_inches='tight') 
    else: 
        print("Did not save figure. Add filename in arguments to be able to save figure.")
    plt.show()
    plt.close('all')