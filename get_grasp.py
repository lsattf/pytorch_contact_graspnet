import os
import yaml
import numpy as np
import torch



def get_bin_vals(global_config):
    """
    Creates bin values for grasping widths according to bounds defined in config

    Arguments:
        global_config {dict} -- config

    Returns:
        tf.constant -- bin value tensor
    """
    bins_bounds = np.array(global_config['DATA']['labels']['offset_bins'])
    if global_config['TEST']['bin_vals'] == 'max':
        bin_vals = (bins_bounds[1:] + bins_bounds[:-1]) / 2
        bin_vals[-1] = bins_bounds[-1]
    elif global_config['TEST']['bin_vals'] == 'mean':
        bin_vals = bins_bounds[1:]
    else:
        raise NotImplementedError

    if not global_config['TEST']['allow_zero_margin']:
        bin_vals = np.minimum(bin_vals, global_config['DATA']['gripper_width'] - global_config['TEST']['extra_opening'])

    torch_bin_vals = torch.tensor(bin_vals, dtype=torch.float32)
    return torch_bin_vals

def build_6d_grasp(approach_dirs, base_dirs, contact_pts, thickness, gripper_depth = 0.1034):
    """
    Build 6-DoF grasps + width from point-wise network predictions

    Arguments:
        approach_dirs {np.ndarray/tf.tensor} -- Nx3 approach direction vectors
        base_dirs {np.ndarray/tf.tensor} -- Nx3 base direction vectors
        contact_pts {np.ndarray/tf.tensor} -- Nx3 contact points
        thickness {np.ndarray/tf.tensor} -- Nx1 grasp width

    Keyword Arguments:
        use_tf {bool} -- whether inputs and outputs are tf tensors (default: {False})
        gripper_depth {float} -- distance from gripper coordinate frame to gripper baseline in m (default: {0.1034})

    Returns:
        np.ndarray -- Nx4x4 grasp poses in camera coordinates
    """
    test1 = torch.sum(base_dirs ** 2, 2)
    test2 = torch.sum(approach_dirs ** 2, 2)
    grasps_R = torch.stack([base_dirs, torch.cross(approach_dirs, base_dirs), approach_dirs], dim=3)
    grasps_t = contact_pts + torch.unsqueeze(thickness, 2) / 2 * base_dirs - gripper_depth * approach_dirs
    ones = torch.ones((contact_pts.shape[0], contact_pts.shape[1], 1, 1), dtype=torch.float32)
    zeros = torch.zeros((contact_pts.shape[0], contact_pts.shape[1], 1, 3), dtype=torch.float32)
    homog_vec = torch.cat((zeros, ones), dim=3)
    grasps = torch.cat((torch.cat((grasps_R, torch.unsqueeze(grasps_t, 3)), dim=3), homog_vec), dim=2)

    # else:
    #     grasps = []
    #     for i in range(len(contact_pts)):
    #         grasp = np.eye(4)
    #         grasp[:3,0] = base_dirs[i] / np.linalg.norm(base_dirs[i])
    #         grasp[:3,2] = approach_dirs[i] / np.linalg.norm(approach_dirs[i])
    #         grasp_y = np.cross( grasp[:3,2],grasp[:3,0])
    #         grasp[:3,1] = grasp_y / np.linalg.norm(grasp_y)
    #         # base_gripper xyz = contact + thickness / 2 * baseline_dir - gripper_d * approach_dir
    #         grasp[:3,3] = contact_pts[i] + thickness[i] / 2 * grasp[:3,0] - gripper_depth * grasp[:3,2]
    #         # grasp[0,3] = finger_width
    #         grasps.append(grasp)
    #     grasps = np.array(grasps)

    return grasps


def recursive_key_value_assign(d, ks, v):
    """
    Recursive value assignment to a nested dict

    Arguments:
        d {dict} -- dict
        ks {list} -- list of hierarchical keys
        v {value} -- value to assign
    """

    if len(ks) > 1:
        recursive_key_value_assign(d[ks[0]], ks[1:], v)
    elif len(ks) == 1:
        d[ks[0]] = v


def load_config(checkpoint_dir, batch_size=None, max_epoch=None, data_path=None, arg_configs=[], save=False):
    """
    Loads yaml config file and overwrites parameters with function arguments and --arg_config parameters

    Arguments:
        checkpoint_dir {str} -- Checkpoint directory where config file was copied to

    Keyword Arguments:
        batch_size {int} -- [description] (default: {None})
        max_epoch {int} -- "epochs" (number of scenes) to train (default: {None})
        data_path {str} -- path to scenes with contact grasp data (default: {None})
        arg_configs {list} -- Overwrite config parameters by hierarchical command line arguments (default: {[]})
        save {bool} -- Save overwritten config file (default: {False})

    Returns:
        [dict] -- Config
    """

    config_path = os.path.join(checkpoint_dir, 'config.yaml')
    config_path = config_path if os.path.exists(config_path) else os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        global_config = yaml.load(f, Loader=yaml.Loader)

    for conf in arg_configs:
        k_str, v = conf.split(':')
        try:
            v = eval(v)
        except:
            pass
        ks = [int(k) if k.isdigit() else k for k in k_str.split('.')]

        recursive_key_value_assign(global_config, ks, v)

    if batch_size is not None:
        global_config['OPTIMIZER']['batch_size'] = int(batch_size)
    if max_epoch is not None:
        global_config['OPTIMIZER']['max_epoch'] = int(max_epoch)
    if data_path is not None:
        global_config['DATA']['data_path'] = data_path

    global_config['DATA']['classes'] = None

    if save:
        with open(os.path.join(checkpoint_dir, 'config.yaml'), 'w') as f:
            yaml.dump(global_config, f)

    return global_config


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

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print('farthest points max distance : {}'.format(np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters


def select_grasps(contact_pts, contact_conf, max_farthest_points=150, num_grasps=200, first_thres=0.25,
                  second_thres=0.2, with_replacement=False):
    """
    Select subset of num_grasps by contact confidence thresholds and farthest contact point sampling.

    1.) Samples max_farthest_points among grasp contacts with conf > first_thres
    2.) Fills up remaining grasp contacts to a maximum of num_grasps with highest confidence contacts with conf > second_thres

    Arguments:
        contact_pts {np.ndarray} -- num_point x 3 subset of input point cloud for which we have predictions
        contact_conf {[type]} -- num_point x 1 confidence of the points being a stable grasp contact

    Keyword Arguments:
        max_farthest_points {int} -- Maximum amount from num_grasps sampled with farthest point sampling (default: {150})
        num_grasps {int} -- Maximum number of grasp proposals to select (default: {200})
        first_thres {float} -- first confidence threshold for farthest point sampling (default: {0.6})
        second_thres {float} -- second confidence threshold for filling up grasp proposals (default: {0.6})
        with_replacement {bool} -- Return fixed number of num_grasps with conf > first_thres and repeat if there are not enough (default: {False})

    Returns:
        [np.ndarray] -- Indices of selected contact_pts
    """
    contact_pts = contact_pts.detach().numpy()
    contact_conf = contact_conf.detach().numpy()
    grasp_conf = contact_conf.squeeze()
    contact_pts = contact_pts.squeeze()

    conf_idcs_greater_than = np.nonzero(grasp_conf > first_thres)[0]
    _, center_indexes = farthest_points(contact_pts[conf_idcs_greater_than, :3],
                                        np.minimum(max_farthest_points, len(conf_idcs_greater_than)),
                                        distance_by_translation_point, return_center_indexes=True)

    remaining_confidences = np.setdiff1d(np.arange(len(grasp_conf)), conf_idcs_greater_than[center_indexes])
    sorted_confidences = np.argsort(grasp_conf)[::-1]
    mask = np.in1d(sorted_confidences, remaining_confidences)
    sorted_remaining_confidence_idcs = sorted_confidences[mask]

    if with_replacement:
        selection_idcs = list(conf_idcs_greater_than[center_indexes])
        j = len(selection_idcs)
        while j < num_grasps and conf_idcs_greater_than.shape[0] > 0:
            selection_idcs.append(conf_idcs_greater_than[j % len(conf_idcs_greater_than)])
            j += 1
        selection_idcs = np.array(selection_idcs)

    else:
        remaining_idcs = sorted_remaining_confidence_idcs[:num_grasps - len(conf_idcs_greater_than[center_indexes])]
        remaining_conf_idcs_greater_than = np.nonzero(grasp_conf[remaining_idcs] > second_thres)[0]
        selection_idcs = np.union1d(conf_idcs_greater_than[center_indexes],
                                    remaining_idcs[remaining_conf_idcs_greater_than])
        selection_idcs = torch.tensor(selection_idcs)
    return selection_idcs


def filter_segment(contact_pts, segment_pc, thres=0.00001):
    """
    Filter grasps to obtain contacts on specified point cloud segment

    :param contact_pts: Nx3 contact points of all grasps in the scene
    :param segment_pc: Mx3 segmented point cloud of the object of interest
    :param thres: maximum distance in m of filtered contact points from segmented point cloud
    :returns: Contact/Grasp indices that lie in the point cloud segment
    """
    contact_pts = contact_pts.detach().numpy()
    filtered_grasp_idcs = np.array([], dtype=np.int32)

    if contact_pts.shape[0] > 0 and segment_pc.shape[0] > 0:
        try:
            dists = contact_pts[:, :3].reshape(-1, 1, 3) - segment_pc.reshape(1, -1, 3)
            min_dists = np.min(np.linalg.norm(dists, axis=2), axis=1)
            filtered_grasp_idcs = np.where(min_dists < thres)
        except:
            pass
    filtered_grasp_idcs = torch.tensor(np.array(filtered_grasp_idcs))
    return filtered_grasp_idcs

def get_grasp_and_score(global_config, end_points, pc_mean, pc_segments, convert_cam_coords=True, filter_grasp=False, k=0, use_end_point_test=False):
    if use_end_point_test:
        end_points['grasp_offset_head'] = torch.tensor(end_points['grasp_offset_head'])
        end_points['approach_dir_head'] = torch.tensor(end_points['approach_dir_head'])
        end_points['grasp_dir_head'] = torch.tensor(end_points['grasp_dir_head'])
        end_points['pred_points'] = torch.tensor(end_points['pred_points'])
        end_points['binary_seg_pred'] = torch.tensor(end_points['binary_seg_pred'])

    torch_bin_vals = get_bin_vals(global_config)
    if global_config['MODEL']['bin_offsets']:
        offset_bin_pred_vals = torch_bin_vals[torch.argmax(end_points['grasp_offset_head'], dim=2)]
    else:
        offset_bin_pred_vals = end_points['grasp_offset_pred'][:, :, 0]

    pred_grasps_cam = build_6d_grasp(end_points['approach_dir_head'], end_points['grasp_dir_head'], end_points['pred_points'], offset_bin_pred_vals)  # b x num_point x 4 x 4

    pred_scores = end_points['binary_seg_pred']
    pred_points = end_points['pred_points']


    pred_grasps_cam = pred_grasps_cam.reshape(-1, *pred_grasps_cam.shape[-2:])
    pred_scores = pred_scores.reshape(-1)
    pred_points = pred_points.reshape(-1, pred_points.shape[-1])
    # uncenter grasps
    pred_grasps_cam[:, :3, 3] += pc_mean.reshape(-1, 3)
    pred_points[:, :3] += pc_mean.reshape(-1, 3)
    selection_idcs = select_grasps(pred_points[:, :3], pred_scores,
                                        global_config['TEST']['max_farthest_points'],
                                        global_config['TEST']['num_samples'],
                                        global_config['TEST']['first_thres'],
                                        global_config['TEST']['second_thres'] if 'second_thres' in global_config['TEST'] else global_config['TEST']['first_thres'],
                                        with_replacement=global_config['TEST']['with_replacement'])

    if not torch.any(selection_idcs):
        selection_idcs = torch.tensor([], dtype=torch.int32)

    if 'center_to_tip' in global_config['TEST'] and global_config['TEST']['center_to_tip']:
        pred_grasps_cam[:, :3, 3] -= pred_grasps_cam[:, :3, 2] * (global_config['TEST']['center_to_tip'] / 2)

    # convert back to opencv coordinates
    if convert_cam_coords:
        pred_grasps_cam[:, :2, :] *= -1
        pred_points[:, :2] *= -1
    pred_grasps_cam = pred_grasps_cam[selection_idcs]
    pred_scores = pred_scores[selection_idcs]


    if filter_grasp:
        segment_idcs = filter_segment(pred_points[selection_idcs].squeeze(), pc_segments[k], thres=global_config['TEST']['filter_thres'])
        return pred_grasps_cam[segment_idcs[0]], pred_scores[segment_idcs[0]]

    return pred_grasps_cam, pred_scores
