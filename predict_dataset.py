import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import cv2
import glob
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
import pickle
import trimesh
import trimesh.transformations as tra
from network import load_config


def load_graspnet_data(rgb_image_path):
    """
    Loads data from the GraspNet-1Billion dataset
    # https://graspnet.net/

    :param rgb_image_path: .png file path to depth image in graspnet dataset
    :returns: (depth, rgb, segmap, K)
    """

    depth = np.array(Image.open(rgb_image_path)) / 1000.  # m to mm
    segmap = np.array(Image.open(rgb_image_path.replace('depth', 'label')))
    rgb = np.array(Image.open(rgb_image_path.replace('depth', 'rgb')))

    # graspnet images are upside down, rotate for inference
    # careful: rotate grasp poses back for evaluation
    depth = np.rot90(depth, 2)
    segmap = np.rot90(segmap, 2)
    rgb = np.rot90(rgb, 2)

    if 'kinect' in rgb_image_path:
        # Kinect azure:
        K = np.array([[631.54864502, 0., 638.43517329],
                      [0., 631.20751953, 366.49904066],
                      [0., 0., 1.]])
    else:
        # Realsense:
        K = np.array([[616.36529541, 0., 310.25881958],
                      [0., 616.20294189, 236.59980774],
                      [0., 0., 1.]])

    return depth, rgb, segmap, K

def load_available_input_data(p, K=None):
    """
    Load available data from input file path.

    Numpy files .npz/.npy should have keys
    'depth' + 'K' + (optionally) 'segmap' + (optionally) 'rgb'
    or for point clouds:
    'xyz' + (optionally) 'xyz_color'

    png files with only depth data (in mm) can be also loaded.
    If the image path is from the GraspNet dataset, corresponding rgb, segmap and intrinic are also loaded.

    :param p: .png/.npz/.npy file path that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: All available data among segmap, rgb, depth, cam_K, pc_full, pc_colors
    """

    segmap, rgb, depth, pc_full, pc_colors = None, None, None, None, None

    if K is not None:
        if isinstance(K, str):
            cam_K = eval(K)
        cam_K = np.array(cam_K).reshape(3,3)

    if '.np' in p:
        data = np.load(p, allow_pickle=True)
        if '.npz' in p:
            keys = data.files
        else:
            keys = []
            if len(data.shape) == 0:
                data = data.item()
                keys = data.keys()
            elif data.shape[-1] == 3:
                pc_full = data
            else:
                depth = data

        if 'depth' in keys:
            depth = data['depth']
            if K is None and 'K' in keys:
                cam_K = data['K'].reshape(3, 3)
            if 'segmap' in keys:
                segmap = data['segmap']
            if 'seg' in keys:
                segmap = data['seg']
            if 'rgb' in keys:
                rgb = data['rgb']
                rgb = np.array(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        elif 'xyz' in keys:
            pc_full = np.array(data['xyz']).reshape(-1, 3)
            if 'xyz_color' in keys:
                pc_colors = data['xyz_color']
    elif '.png' in p:
        if os.path.exists(p.replace('depth', 'label')):
            # graspnet data
            depth, rgb, segmap, K = load_graspnet_data(p)
        elif os.path.exists(p.replace('depths', 'images').replace('npy', 'png')):
            rgb = np.array(Image.open(p.replace('depths', 'images').replace('npy', 'png')))
        else:
            depth = np.array(Image.open(p))
    else:
        raise ValueError('{} is neither png nor npz/npy file'.format(p))

    return segmap, rgb, depth, cam_K, pc_full, pc_colors

def extract_point_clouds(depth, K, segmap=None, rgb=None, z_range=[0.2, 1.8], segmap_id=0,
                             skip_border_objects=False, margin_px=5):
    """
    Converts depth map + intrinsics to point cloud.
    If segmap is given, also returns segmented point clouds. If rgb is given, also returns pc_colors.

    Arguments:
        depth {np.ndarray} -- HxW depth map in m
        K {np.ndarray} -- 3x3 camera Matrix

    Keyword Arguments:
        segmap {np.ndarray} -- HxW integer array that describes segeents (default: {None})
        rgb {np.ndarray} -- HxW rgb image (default: {None})
        z_range {list} -- Clip point cloud at minimum/maximum z distance (default: {[0.2,1.8]})
        segmap_id {int} -- Only return point cloud segment for the defined id (default: {0})
        skip_border_objects {bool} -- Skip segments that are at the border of the depth map to avoid artificial edges (default: {False})
        margin_px {int} -- Pixel margin of skip_border_objects (default: {5})

    Returns:
        [np.ndarray, dict[int:np.ndarray], np.ndarray] -- Full point cloud, point cloud segments, point cloud colors
    """

    if K is None:
        raise ValueError('K is required either as argument --K or from the input numpy file')

    # Convert to pc
    pc_full, pc_colors = depth2pc(depth, K, rgb)

    # Threshold distance
    if pc_colors is not None:
        pc_colors = pc_colors[(pc_full[:, 2] < z_range[1]) & (pc_full[:, 2] > z_range[0])]
    pc_full = pc_full[(pc_full[:, 2] < z_range[1]) & (pc_full[:, 2] > z_range[0])]

    # Extract instance point clouds from segmap and depth map
    pc_segments = {}
    if segmap is not None:
        pc_segments = {}
        obj_instances = [segmap_id] if segmap_id else np.unique(segmap[segmap > 0])
        for i in obj_instances:
            if skip_border_objects and not i == segmap_id:
                obj_i_y, obj_i_x = np.where(segmap == i)
                if np.any(obj_i_x < margin_px) or np.any(obj_i_x > segmap.shape[1] - margin_px) or np.any(
                        obj_i_y < margin_px) or np.any(obj_i_y > segmap.shape[0] - margin_px):
                    print('object {} not entirely in image bounds, skipping'.format(i))
                    continue
            inst_mask = segmap == i
            pc_segment, _ = depth2pc(depth * inst_mask, K) # element-wise * of two array
            pc_segments[i] = pc_segment[(pc_segment[:, 2] < z_range[1]) & (pc_segment[:, 2] > z_range[
                0])]  # regularize_pc_point_count(pc_segment, grasp_estimator._contact_grasp_cfg['DATA']['num_point'])

    return pc_full, pc_segments, pc_colors


def depth2pc(depth, K, rgb=None):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """

    mask = np.where(depth > 0)
    x, y = mask[1], mask[0]

    normalized_x = (x.astype(np.float32) - K[0, 2])
    normalized_y = (y.astype(np.float32) - K[1, 2])

    world_x = normalized_x * depth[y, x] / K[0, 0]
    world_y = normalized_y * depth[y, x] / K[1, 1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y, x, :]

    pc = np.vstack((world_x, world_y, world_z)).T
    return (pc, rgb)

def reject_median_outliers(data, m=0.4, z_only=False):
    """
    Reject outliers with median absolute distance m

    Arguments:
        data {[np.ndarray]} -- Numpy array such as point cloud

    Keyword Arguments:
        m {[float]} -- Maximum absolute distance from median in m (default: {0.4})
        z_only {[bool]} -- filter only via z_component (default: {False})

    Returns:
        [np.ndarray] -- Filtered data without outliers
    """
    if z_only:
        d = np.abs(data[:,2:3] - np.median(data[:,2:3]))
    else:
        d = np.abs(data - np.median(data, axis=0, keepdims=True))

    return data[np.sum(d, axis=1) < m]


def distance_by_translation_point(p1, p2):
    """
      Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def farthest_points(data, nclusters, dist_func, return_center_indexes=False, return_distances=False, verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of
          clusters.
        return_distances: bool, If True, return distances of each point from centers.

      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0], dtype=np.int32), np.arange(data.shape[0], dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0],), dtype=np.int32) * -1
    distances = np.ones((data.shape[0],), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1
        #broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        broadcasted_data = np.tile(data[index], shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        if iter == 1:
            print('new distance shape:', distances.shape)
        clusters[distances == new_distances] = iter
        if verbose:
            print('farthest points max distance : {}'.format(np.max(distances)))
    print('test clusters')
    print(clusters.shape)
    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates

      :param pc: Nx3 point cloud
      :param npoints: number of points the regularized point cloud should have
      :param use_farthest_point: use farthest point sampling to downsample the points, runs slower.
      :returns: npointsx3 regularized point cloud
    """

    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(pc, npoints, distance_by_translation_point, return_center_indexes=True)
        else:
            center_indexes = np.random.choice(range(pc.shape[0]), size=npoints, replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc

def extract_3d_cam_boxes(full_pc, pc_segments, min_size=0.3, max_size=0.6):
    """
    Extract 3D bounding boxes around the pc_segments for inference to create
    dense and zoomed-in predictions but still take context into account.

    :param full_pc: Nx3 scene point cloud
    :param pc_segments: Mx3 segmented point cloud of the object of interest
    :param min_size: minimum side length of the 3D bounding box
    :param max_size: maximum side length of the 3D bounding box
    :returns: (pc_regions, obj_centers) Point cloud box regions and their centers
    """

    pc_regions = {}
    obj_centers = {}

    for i in pc_segments:
        pc_segments[i] = reject_median_outliers(pc_segments[i], m=0.4, z_only=False)

        if np.any(pc_segments[i]):
            max_bounds = np.max(pc_segments[i][:, :3], axis=0)
            min_bounds = np.min(pc_segments[i][:, :3], axis=0)

            obj_extent = max_bounds - min_bounds
            obj_center = min_bounds + obj_extent / 2

            # cube size is between 0.3 and 0.6 depending on object extents
            size = np.minimum(np.maximum(np.max(obj_extent) * 2, min_size), max_size)
            print('Extracted Region Cube Size: ', size)
            partial_pc = full_pc[
                np.all(full_pc > (obj_center - size / 2), axis=1) & np.all(full_pc < (obj_center + size / 2),
                                                                           axis=1)]
            if np.any(partial_pc):
                # change to use load config file
                partial_pc = regularize_pc_point_count(partial_pc,
                                                       20000,
                                                       use_farthest_point=False)
                pc_regions[i] = partial_pc
                obj_centers[i] = obj_center

    return pc_regions, obj_centers


def preprocess_pc_for_inference(input_pc, num_point, pc_mean=None, return_mean=False, use_farthest_point=False,
                                convert_to_internal_coords=False):
    """
    Various preprocessing of the point cloud (downsampling, centering, coordinate transforms)

    Arguments:
        input_pc {np.ndarray} -- Nx3 input point cloud
        num_point {int} -- downsample to this amount of points

    Keyword Arguments:
        pc_mean {np.ndarray} -- use 3x1 pre-computed mean of point cloud  (default: {None})
        return_mean {bool} -- whether to return the point cloud mean (default: {False})
        use_farthest_point {bool} -- use farthest point for downsampling (slow and suspectible to outliers) (default: {False})
        convert_to_internal_coords {bool} -- Convert from opencv to internal coordinates (x left, y up, z front) (default: {False})

    Returns:
        [np.ndarray] -- num_pointx3 preprocessed point cloud
    """
    normalize_pc_count = input_pc.shape[0] != num_point
    if normalize_pc_count:
        pc = regularize_pc_point_count(input_pc, num_point, use_farthest_point=use_farthest_point).copy()
    else:
        pc = input_pc.copy()

    if convert_to_internal_coords:
        pc[:, :2] *= -1

    if pc_mean is None:
        pc_mean = np.mean(pc, 0)

    pc -= np.expand_dims(pc_mean, 0)
    if return_mean:
        return pc, pc_mean
    else:
        return pc


def show_image(rgb, segmap):
    """
    Overlay rgb image with segmentation and imshow segment

    Arguments:
        rgb {np.ndarray} -- color image
        segmap {np.ndarray} -- integer segmap of same size as rgb
    """
    plt.figure()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    plt.ion()
    plt.show()

    if rgb is not None:
        plt.imshow(rgb)
    if segmap is not None:
        cmap = plt.get_cmap('rainbow')
        cmap.set_under(alpha=0.0)
        plt.imshow(segmap, cmap=cmap, alpha=0.5, vmin=0.0001)
    plt.draw()
    plt.pause(0.001)


def visualize_grasps(full_pc, pred_grasps_cam, scores, plot_opencv_cam=False, pc_colors=None, gripper_openings=None,
                     gripper_width=0.08):
    """Visualizes colored point cloud and predicted grasps. If given, colors grasps by segmap regions.
    Thick grasp is most confident per segment. For scene point cloud predictions, colors grasps according to confidence.

    Arguments:
        full_pc {np.ndarray} -- Nx3 point cloud of the scene
        pred_grasps_cam {dict[int:np.ndarray]} -- Predicted 4x4 grasp trafos per segment or for whole point cloud
        scores {dict[int:np.ndarray]} -- Confidence scores for grasps

    Keyword Arguments:
        plot_opencv_cam {bool} -- plot camera coordinate frame (default: {False})
        pc_colors {np.ndarray} -- Nx3 point cloud colors (default: {None})
        gripper_openings {dict[int:np.ndarray]} -- Predicted grasp widths (default: {None})
        gripper_width {float} -- If gripper_openings is None, plot grasp widths (default: {0.008})
    """

    print('Visualizing...takes time')
    cm = plt.get_cmap('rainbow')
    cm2 = plt.get_cmap('gist_rainbow')

    fig = mlab.figure('Pred Grasps')
    mlab.view(azimuth=180, elevation=180, distance=0.2)
    draw_pc_with_colors(full_pc, pc_colors)
    colors = [cm(1. * i / len(pred_grasps_cam))[:3] for i in range(len(pred_grasps_cam))]
    colors2 = {k: cm2(0.5 * np.max(scores[k]))[:3] for k in pred_grasps_cam if np.any(pred_grasps_cam[k])}

    if plot_opencv_cam:
        plot_coordinates(np.zeros(3, ), np.eye(3, 3))
    for i, k in enumerate(pred_grasps_cam):
        if np.any(pred_grasps_cam[k]):
            gripper_openings_k = np.ones(len(pred_grasps_cam[k])) * gripper_width if gripper_openings is None else \
            gripper_openings[k]
            if len(pred_grasps_cam) > 1:
                draw_grasps(pred_grasps_cam[k], np.eye(4), color=colors[i], gripper_openings=gripper_openings_k)
                draw_grasps([pred_grasps_cam[k][np.argmax(scores[k])]], np.eye(4), color=colors2[k],
                            gripper_openings=[gripper_openings_k[np.argmax(scores[k])]], tube_radius=0.0025)
            else:
                colors3 = [cm2(0.5 * score)[:3] for score in scores[k]]
                draw_grasps(pred_grasps_cam[k], np.eye(4), colors=colors3, gripper_openings=gripper_openings_k)
    mlab.show()


def draw_pc_with_colors(pc, pc_colors=None, single_color=(0.3, 0.3, 0.3), mode='2dsquare', scale_factor=0.0018):
    """
    Draws colored point clouds

    Arguments:
        pc {np.ndarray} -- Nx3 point cloud
        pc_colors {np.ndarray} -- Nx3 point cloud colors

    Keyword Arguments:
        single_color {tuple} -- single color for point cloud (default: {(0.3,0.3,0.3)})
        mode {str} -- primitive type to plot (default: {'point'})
        scale_factor {float} -- Scale of primitives. Does not work for points. (default: {0.002})

    """

    if pc_colors is None:
        mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=single_color, scale_factor=scale_factor, mode=mode)
    else:
        # create direct grid as 256**3 x 4 array
        def create_8bit_rgb_lut():
            xl = np.mgrid[0:256, 0:256, 0:256]
            lut = np.vstack((xl[0].reshape(1, 256 ** 3),
                             xl[1].reshape(1, 256 ** 3),
                             xl[2].reshape(1, 256 ** 3),
                             255 * np.ones((1, 256 ** 3)))).T
            return lut.astype('int32')

        scalars = pc_colors[:, 0] * 256 ** 2 + pc_colors[:, 1] * 256 + pc_colors[:, 2]
        rgb_lut = create_8bit_rgb_lut()
        points_mlab = mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], scalars, mode=mode, scale_factor=.0018)
        points_mlab.glyph.scale_mode = 'scale_by_vector'
        points_mlab.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, rgb_lut.shape[0])
        points_mlab.module_manager.scalar_lut_manager.lut.number_of_colors = rgb_lut.shape[0]
        points_mlab.module_manager.scalar_lut_manager.lut.table = rgb_lut


def draw_grasps(grasps, cam_pose, gripper_openings, color=(0, 1., 0), colors=None, show_gripper_mesh=False,
                tube_radius=0.0008):
    """
    Draws wireframe grasps from given camera pose and with given gripper openings

    Arguments:
        grasps {np.ndarray} -- Nx4x4 grasp pose transformations
        cam_pose {np.ndarray} -- 4x4 camera pose transformation
        gripper_openings {np.ndarray} -- Nx1 gripper openings

    Keyword Arguments:
        color {tuple} -- color of all grasps (default: {(0,1.,0)})
        colors {np.ndarray} -- Nx3 color of each grasp (default: {None})
        tube_radius {float} -- Radius of the grasp wireframes (default: {0.0008})
        show_gripper_mesh {bool} -- Renders the gripper mesh for one of the grasp poses (default: {False})
    """

    gripper = create_gripper('panda')
    gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    mid_point = 0.5 * (gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3],
                                gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])

    if show_gripper_mesh and len(grasps) > 0:
        plot_mesh(gripper.hand, cam_pose, grasps[0])

    all_pts = []
    connections = []
    index = 0
    N = 7
    for i, (g, g_opening) in enumerate(zip(grasps, gripper_openings)):
        gripper_control_points_closed = grasp_line_plot.copy()
        gripper_control_points_closed[2:, 0] = np.sign(grasp_line_plot[2:, 0]) * g_opening / 2
        # if i == 0:
        #         print('test matrix------------------------------------')
        #         print(g)
        #         print(g[:3, :3].T)
        #         print('test matrix------------------------------------')
        pts = np.matmul(gripper_control_points_closed, g[:3, :3].T)
        test1 = np.sum(g[0, :3]**2)
        test2 = np.sum(g[1, :3]**2)
        test3 = np.sum(g[2, :3]**2)
        # if i == 0:
        #         print('test pts------------------------------------')
        #         print(gripper_control_points_closed)
        #         print('test pts------------------------------------')
        pts += np.expand_dims(g[:3, 3], 0)
        pts_homog = np.concatenate((pts, np.ones((7, 1))), axis=1)
        pts = np.dot(pts_homog, cam_pose.T)[:, :3]

        color = color if colors is None else colors[i]

        all_pts.append(pts)
        connections.append(np.vstack([np.arange(index, index + N - 1.5),
                                      np.arange(index + 1, index + N - .5)]).T)
        index += N
        # mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2], color=color, tube_radius=tube_radius, opacity=1.0)

    # speeds up plot3d because only one vtk object
    all_pts = np.vstack(all_pts)
    connections = np.vstack(connections)
    src = mlab.pipeline.scalar_scatter(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2])
    src.mlab_source.dataset.lines = connections
    src.update()
    lines = mlab.pipeline.tube(src, tube_radius=tube_radius, tube_sides=12)
    mlab.pipeline.surface(lines, color=color, opacity=1.0)


def plot_mesh(mesh, cam_trafo=np.eye(4), mesh_pose=np.eye(4)):
    """
    Plots mesh in mesh_pose from

    Arguments:
        mesh {trimesh.base.Trimesh} -- input mesh, e.g. gripper

    Keyword Arguments:
        cam_trafo {np.ndarray} -- 4x4 transformation from world to camera coords (default: {np.eye(4)})
        mesh_pose {np.ndarray} -- 4x4 transformation from mesh to world coords (default: {np.eye(4)})
    """

    homog_mesh_vert = np.pad(mesh.vertices, (0, 1), 'constant', constant_values=(0, 1))
    mesh_cam = homog_mesh_vert.dot(mesh_pose.T).dot(cam_trafo.T)[:, :3]
    mlab.triangular_mesh(mesh_cam[:, 0],
                         mesh_cam[:, 1],
                         mesh_cam[:, 2],
                         mesh.faces,
                         colormap='Blues',
                         opacity=0.5)


def plot_coordinates(t, r, tube_radius=0.005):
    """
    plots coordinate frame

    Arguments:
        t {np.ndarray} -- translation vector
        r {np.ndarray} -- rotation matrix

    Keyword Arguments:
        tube_radius {float} -- radius of the plotted tubes (default: {0.005})
    """
    mlab.plot3d([t[0], t[0] + 0.2 * r[0, 0]], [t[1], t[1] + 0.2 * r[1, 0]], [t[2], t[2] + 0.2 * r[2, 0]],
                color=(1, 0, 0), tube_radius=tube_radius, opacity=1)
    mlab.plot3d([t[0], t[0] + 0.2 * r[0, 1]], [t[1], t[1] + 0.2 * r[1, 1]], [t[2], t[2] + 0.2 * r[2, 1]],
                color=(0, 1, 0), tube_radius=tube_radius, opacity=1)
    mlab.plot3d([t[0], t[0] + 0.2 * r[0, 2]], [t[1], t[1] + 0.2 * r[1, 2]], [t[2], t[2] + 0.2 * r[2, 2]],
                color=(0, 0, 1), tube_radius=tube_radius, opacity=1)


class PandaGripper(object):
    """An object representing a Franka Panda gripper."""

    def __init__(self, q=None, num_contact_points_per_finger=10,
                 root_folder=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        """Create a Franka Panda parallel-yaw gripper object.

        Keyword Arguments:
            q {list of int} -- opening configuration (default: {None})
            num_contact_points_per_finger {int} -- contact points per finger (default: {10})
            root_folder {str} -- base folder for model files (default: {''})
        """
        self.joint_limits = [0.0, 0.04]
        self.root_folder = root_folder

        self.default_pregrasp_configuration = 0.04
        if q is None:
            q = self.default_pregrasp_configuration

        self.q = q
        fn_base = os.path.join(root_folder, 'gripper_models/panda_gripper/hand.stl')
        fn_finger = os.path.join(root_folder, 'gripper_models/panda_gripper/finger.stl')

        self.base = trimesh.load(fn_base)
        self.finger_l = trimesh.load(fn_finger)
        self.finger_r = self.finger_l.copy()

        # transform fingers relative to the base
        self.finger_l.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_l.apply_translation([+q, 0, 0.0584])
        self.finger_r.apply_translation([-q, 0, 0.0584])

        self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])

        self.contact_ray_origins = []
        self.contact_ray_directions = []

        # coords_path = os.path.join(root_folder, 'gripper_control_points/panda_gripper_coords.npy')
        with open(os.path.join(root_folder, 'gripper_control_points/panda_gripper_coords.pickle'), 'rb') as f:
            self.finger_coords = pickle.load(f, encoding='latin1')
        finger_direction = self.finger_coords['gripper_right_center_flat'] - self.finger_coords[
            'gripper_left_center_flat']
        self.contact_ray_origins.append(np.r_[self.finger_coords['gripper_left_center_flat'], 1])
        self.contact_ray_origins.append(np.r_[self.finger_coords['gripper_right_center_flat'], 1])
        self.contact_ray_directions.append(finger_direction / np.linalg.norm(finger_direction))
        self.contact_ray_directions.append(-finger_direction / np.linalg.norm(finger_direction))

        self.contact_ray_origins = np.array(self.contact_ray_origins)
        self.contact_ray_directions = np.array(self.contact_ray_directions)

    def get_meshes(self):
        """Get list of meshes that this gripper consists of.

        Returns:
            list of trimesh -- visual meshes
        """
        return [self.finger_l, self.finger_r, self.base]

    def get_closing_rays_contact(self, transform):
        """Get an array of rays defining the contact locations and directions on the hand.

        Arguments:
            transform {[nump.array]} -- a 4x4 homogeneous matrix
            contact_ray_origin {[nump.array]} -- a 4x1 homogeneous vector
            contact_ray_direction {[nump.array]} -- a 4x1 homogeneous vector

        Returns:
            numpy.array -- transformed rays (origin and direction)
        """
        return transform[:3, :].dot(
            self.contact_ray_origins.T).T, transform[:3, :3].dot(self.contact_ray_directions.T).T

    def get_control_point_tensor(self, batch_size, symmetric=False, convex_hull=True):
        """
        Outputs a 5 point gripper representation of shape (batch_size x 5 x 3).

        Arguments:
            batch_size {int} -- batch size

        Keyword Arguments:
            use_tf {bool} -- outputing a tf tensor instead of a numpy array (default: {True})
            symmetric {bool} -- Output the symmetric control point configuration of the gripper (default: {False})
            convex_hull {bool} -- Return control points according to the convex hull panda gripper model (default: {True})

        Returns:
            np.ndarray -- control points of the panda gripper
        """

        control_points = np.load(os.path.join(self.root_folder, 'gripper_control_points/panda.npy'))[:, :3]
        if symmetric:
            control_points = [[0, 0, 0], control_points[1, :], control_points[0, :], control_points[-1, :],
                              control_points[-2, :]]
        else:
            control_points = [[0, 0, 0], control_points[0, :], control_points[1, :], control_points[-2, :],
                              control_points[-1, :]]

        control_points = np.asarray(control_points, dtype=np.float32)
        if not convex_hull:
            # actual depth of the gripper different from convex collision model
            control_points[1:3, 2] = 0.0584
        control_points = np.tile(np.expand_dims(control_points, 0), [batch_size, 1, 1])


        return control_points


def create_gripper(name, configuration=None, root_folder=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
    """Create a gripper object.

    Arguments:
        name {str} -- name of the gripper

    Keyword Arguments:
        configuration {list of float} -- configuration (default: {None})
        root_folder {str} -- base folder for model files (default: {''})

    Raises:
        Exception: If the gripper name is unknown.

    Returns:
        [type] -- gripper object
    """
    if name.lower() == 'panda':
        return PandaGripper(q=configuration, root_folder=root_folder)
    else:
        raise Exception("Unknown gripper: {}".format(name))




class ContactDataset(Dataset):
    def __init__(self, input_paths='test_data/0.npy', model_path='finished_model',data_path = '../../contact_graspnet/checkpoints/acronym', K=None, local_regions=True, skip_border_objects=False, filter_grasps=True, z_range=[0.2,1.8], forward_passes=1, is_training=False):
        self.pc_mean = []
        self.rgb = []
        self.segmap = []
        self.pc_full = []
        self.pc_colors = []
        self.pc_segments = {}
        self.filter_grasps = filter_grasps
        self.cfg = load_config(model_path, data_path=data_path, save=True)
        self.is_training = is_training

        if self.is_training == False:
            for p in glob.glob(input_paths):
                print('Loading ', p)

                pc_segments = {}
                segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p, K=K)

                if segmap is None and (local_regions or filter_grasps):
                    raise ValueError('Need segmentation map to extract local regions or filter grasps')

                if pc_full is None:
                    print('Converting depth to point cloud(s)...')
                    pc_full, pc_segments, pc_colors = extract_point_clouds(depth, cam_K, segmap=segmap,
                                                                           rgb=rgb,
                                                                           skip_border_objects=skip_border_objects,
                                                                           z_range=z_range)
                self.rgb.append(rgb)
                self.segmap.append(segmap)
                self.pc_colors.append(pc_colors)
                # Predict grasps in local regions or full pc
                pc_batch = {}
                pc_mean = {}
                if local_regions:
                    pc_regions, _ = extract_3d_cam_boxes(pc_full, pc_segments)
                    for k, pc_region in pc_regions.items():
                        pc, pc_mean[k] = preprocess_pc_for_inference(pc_region.squeeze(), 20000, return_mean=True,
                                                                     convert_to_internal_coords=True)  # change to use load config file
                        if len(pc.shape) == 2:
                            pc_batch[k] = pc[np.newaxis, :, :]

                        if forward_passes > 1:
                            pc_batch[k] = np.tile(pc_batch[k], (forward_passes, 1, 1))


                else:
                    pc_full = regularize_pc_point_count(pc_full, 20000)  # change to use load config file
                    pc, pc_mean[-1] = preprocess_pc_for_inference(pc_full.squeeze(), 20000, return_mean=True,
                                                                  convert_to_internal_coords=True)  # change to use load config file
                    if len(pc.shape) == 2:
                        pc_batch[-1] = pc[np.newaxis, :, :]

                    if forward_passes > 1:
                        pc_batch[-1] = np.tile(pc_batch, (forward_passes, 1, 1))
                    # print('Generated {} grasps'.format(len(pred_grasps_cam[-1])))

                self.pc_batch = pc_batch
                self.pc_full.append(pc_full)
                self.pc_mean.append(pc_mean)
                self.pc_segments = pc_segments



    def __len__(self):
        if self.is_training == False:
            return len(list(self.pc_batch.keys())),
        else:
            pass

    def __getitem__(self, index):
        return self.pc_batch[index]

    def getrgb(self):
        return self.rgb

    def getsegmap(self):
        return self.segmap

    def getpc_full(self):
        return self.pc_full

    def getpc_colors(self):
        return self.pc_colors

    def getpc_mean(self):
        return self.pc_mean

    def getpc_segments(self):
        return self.pc_segments

