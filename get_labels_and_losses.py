import numpy as np
import torch
import torch.nn.functional as F
from pointnet2.models.pointnet2_utils import index_points
from network import get_bin_vals, build_6d_grasp
import mesh_utils


def load_contact_grasps(contact_list, data_config):
    """
    Loads fixed amount of contact grasp data per scene into tf CPU/GPU memory

    Arguments:
        contact_infos {list(dicts)} -- Per scene mesh: grasp contact information
        data_config {dict} -- data config

    Returns:
        [tf_pos_contact_points, tf_pos_contact_dirs, tf_pos_contact_offsets,
        tf_pos_contact_approaches, tf_pos_finger_diffs, tf_scene_idcs,
        all_obj_paths, all_obj_transforms] -- tf.constants with per scene grasp data, object paths/transforms in scene
    """

    num_pos_contacts = data_config['labels']['num_pos_contacts']  # 8000

    pos_contact_points = []
    pos_contact_dirs = []
    pos_finger_diffs = []
    pos_approach_dirs = []

    for i, c in enumerate(contact_list):
        contact_directions_01 = c['scene_contact_points'][:, 0, :] - c['scene_contact_points'][:, 1,
                                                                     :]  # two pairs of points
        all_contact_points = c['scene_contact_points'].reshape(-1, 3)
        all_finger_diffs = np.maximum(np.linalg.norm(contact_directions_01, axis=1),
                                      np.finfo(np.float32).eps)  # two points distance, shape = [point pairs]
        all_contact_directions = np.empty((contact_directions_01.shape[0] * 2, contact_directions_01.shape[1],))
        all_contact_directions[0::2] = -contact_directions_01 / all_finger_diffs[:,
                                                                np.newaxis]  # unit vextor of the contact direction
        all_contact_directions[1::2] = contact_directions_01 / all_finger_diffs[:,
                                                               np.newaxis]  # unit vextor of the reverse contact direction
        all_contact_suc = np.ones_like(all_contact_points[:, 0])  # one vector to accommodate all points success
        all_grasp_transform = c['grasp_transforms'].reshape(-1, 4, 4)
        all_approach_directions = all_grasp_transform[:, :3, 2]  # z axis of the gripper frame

        pos_idcs = np.where(all_contact_suc > 0)[0]  # tuple for where, [0] to get array
        if len(pos_idcs) == 0:
            continue
        # print('total positive contact points ', len(pos_idcs))

        all_pos_contact_points = all_contact_points[pos_idcs]
        all_pos_finger_diffs = all_finger_diffs[
            pos_idcs // 2]  # // floor division 15//2 = floor(7.5)= 7, two paired points with the same diffs
        all_pos_contact_dirs = all_contact_directions[pos_idcs]  # two pair points with reverse contact directions
        all_pos_approach_dirs = all_approach_directions[
            pos_idcs // 2]  # two paired points with the same approach directions

        # Use all positive contacts then mesh_utils with replacement
        if num_pos_contacts > len(all_pos_contact_points) / 2:
            pos_sampled_contact_idcs = np.arange(len(all_pos_contact_points))  # [0 1 2 ... number_of_points-1]
            pos_sampled_contact_idcs_replacement = np.random.choice(np.arange(len(all_pos_contact_points)),
                                                                    num_pos_contacts * 2 - len(all_pos_contact_points),
                                                                    replace=True)
            pos_sampled_contact_idcs = np.hstack((pos_sampled_contact_idcs, pos_sampled_contact_idcs_replacement))
        else:
            pos_sampled_contact_idcs = np.random.choice(np.arange(len(all_pos_contact_points)), num_pos_contacts * 2,
                                                        replace=False)

        pos_contact_points.append(all_pos_contact_points[pos_sampled_contact_idcs,
                                  :])  # list that every element shape = (number_of_sampled_points, 3) (points may contain repeat samples)
        pos_contact_dirs.append(all_pos_contact_dirs[pos_sampled_contact_idcs,
                                :])  # list that every element shape = (number_of_sampled_points, 3( (points may contain repeat samples)
        pos_finger_diffs.append(all_pos_finger_diffs[
                                    pos_sampled_contact_idcs])  # list that every element shape = (number_of_sampled_points, ) (points may contain repeat samples)
        pos_approach_dirs.append(all_pos_approach_dirs[
                                     pos_sampled_contact_idcs])  # list that every element shape = (number_of_sampled_points, 3) (points may contain repeat samples)

    # device = "/cpu:0" if 'to_gpu' in data_config['labels'] and not data_config['labels']['to_gpu'] else "/gpu:0"
    # print("grasp label device: ", device)
    # scene_idcs = torch.tensor(np.arange(0, len(pos_contact_points)), dtype=torch.int32)  # shape = (number_of_files, )
    # pos_contact_points = torch.tensor(np.array(pos_contact_points), dtype=torch.float32)  # shape = (number_of_files, number_of_sampled_points, 3)
    # pos_contact_dirs = F.normalize(torch.tensor(np.array(pos_contact_dirs), dtype=torch.float32), dim=2)  # shape = (number_of_files, number_of_sampled_points, 3) useless normalization
    # pos_finger_diffs = torch.tensor(np.array(pos_finger_diffs), dtype=torch.float32) # shape = (number_of_files, number_of_sampled_points)
    # pos_contact_approaches = F.normalize(torch.tensor(np.array(pos_approach_dirs), dtype=torch.float32), dim=2)  # shape = (number_of_files, number_of_sampled_points, 3) useless normalization

    scene_idcs = torch.tensor(np.arange(0, len(pos_contact_points)), dtype=torch.int32)  # shape = (number_of_files, )
    pos_contact_points = torch.tensor(np.array(pos_contact_points), dtype=torch.float32)  # shape = (number_of_files, number_of_sampled_points, 3)
    pos_contact_dirs = F.normalize(torch.tensor(np.array(pos_contact_dirs), dtype=torch.float32), dim=2)  # shape = (number_of_files, number_of_sampled_points, 3) useless normalization
    pos_finger_diffs = torch.tensor(np.array(pos_finger_diffs), dtype=torch.float32)  # shape = (number_of_files, number_of_sampled_points)
    pos_contact_approaches = F.normalize(torch.tensor(np.array(pos_approach_dirs), dtype=torch.float32), dim=2)  # shape = (number_of_files, number_of_sampled_points, 3) useless normalization

    return pos_contact_points, pos_contact_dirs, pos_contact_approaches, pos_finger_diffs, scene_idcs


def compute_labels(pos_contact_pts_mesh, pos_contact_dirs_mesh, pos_contact_approaches_mesh, pos_finger_diffs,
                   pc_cam_pl, camera_pose_pl, global_config, device):
    """
    Project grasp labels defined on meshes onto rendered point cloud from a camera pose via nearest neighbor contacts within a maximum radius.
    All points without nearby successful grasp contacts are considered negativ contact points.

    Arguments:
        pos_contact_pts_mesh {tf.constant} -- positive contact points on the mesh scene (Mx3)
        pos_contact_dirs_mesh {tf.constant} -- respective contact base directions in the mesh scene (Mx3)
        pos_contact_approaches_mesh {tf.constant} -- respective contact approach directions in the mesh scene (Mx3)
        pos_finger_diffs {tf.constant} -- respective grasp widths in the mesh scene (Mx1)
        pc_cam_pl {tf.placeholder} -- bxNx3 rendered point clouds
        camera_pose_pl {tf.placeholder} -- bx4x4 camera poses
        global_config {dict} -- global config

    Returns:
        [dir_labels_pc_cam, offset_labels_pc, grasp_success_labels_pc, approach_labels_pc_cam] -- Per-point contact success labels and per-contact pose labels in rendered point cloud
    """
    label_config = global_config['DATA']['labels']
    model_config = global_config['MODEL']

    nsample = label_config['k']
    radius = label_config['max_radius']
    filter_z = label_config['filter_z']
    z_val = label_config['z_val']

    camera_pose_pl = torch.tensor(camera_pose_pl).to(device)
    xyz_cam = pc_cam_pl[:, :, :3]
    pad_homog = torch.ones((xyz_cam.shape[0], xyz_cam.shape[1], 1)).to(device)
    pc_mesh = torch.matmul(torch.cat([xyz_cam, pad_homog], dim=2),
                           torch.transpose(torch.linalg.inv(camera_pose_pl), dim0=1, dim1=2))[:, :, :3]

    contact_point_offsets_batch = torch.unsqueeze(pos_finger_diffs, 0).repeat(pc_mesh.shape[0], 1)

    pad_homog2 = torch.ones((pc_mesh.shape[0], pos_contact_dirs_mesh.shape[0], 1)).to(device)
    contact_point_dirs_batch = torch.unsqueeze(pos_contact_dirs_mesh, 0).repeat(pc_mesh.shape[0], 1, 1)
    contact_point_dirs_batch_cam = torch.matmul(contact_point_dirs_batch,
                                                torch.transpose(camera_pose_pl[:, :3, :3], dim0=1, dim1=2))[:, :, :3]

    pos_contact_approaches_batch = torch.unsqueeze(pos_contact_approaches_mesh, 0).repeat(pc_mesh.shape[0], 1, 1)

    pos_contact_approaches_batch_cam = torch.matmul(pos_contact_approaches_batch,
                                                    torch.transpose(camera_pose_pl[:, :3, :3], dim0=1, dim1=2))[:, :,
                                       :3]

    contact_point_batch_mesh = torch.unsqueeze(pos_contact_pts_mesh, 0).repeat(pc_mesh.shape[0], 1, 1)

    contact_point_batch_cam = torch.matmul(torch.cat([contact_point_batch_mesh, pad_homog2], 2),
                                           torch.transpose(camera_pose_pl, dim0=1, dim1=2))[:, :, :3]

    if filter_z:
        dir_filter_passed = torch.gt(contact_point_dirs_batch_cam[:, :, 2:3], z_val).repeat(1, 1, 3)
        contact_point_batch_mesh = torch.where(dir_filter_passed, contact_point_batch_mesh,
                                               torch.ones_like(contact_point_batch_mesh) * 100000)

    squared_dists_all = torch.sum((torch.unsqueeze(contact_point_batch_cam, 1) - torch.unsqueeze(xyz_cam, 2)) ** 2,
                                  dim=3)
    neg_squared_dists_k, close_contact_pt_idcs = torch.topk(-squared_dists_all, nsample, sorted=False)
    squared_dists_k = -neg_squared_dists_k

    # Nearest neighbor mapping
    grasp_success_labels_pc = torch.lt(torch.mean(squared_dists_k, dim=2), radius * radius).type(
        torch.float32)  # (batch_size, num_point)

    grouped_dirs_pc_cam = index_points(contact_point_dirs_batch_cam, close_contact_pt_idcs)
    grouped_approaches_pc_cam = index_points(pos_contact_approaches_batch_cam, close_contact_pt_idcs)
    grouped_offsets = index_points(torch.unsqueeze(contact_point_offsets_batch, 2), close_contact_pt_idcs)

    dir_labels_pc_cam = F.normalize(torch.mean(grouped_dirs_pc_cam, dim=2), dim=2)  # (batch_size, num_point, 3)
    approach_labels_pc_cam = F.normalize(torch.mean(grouped_approaches_pc_cam, dim=2),
                                         dim=2)  # (batch_size, num_point, 3)
    offset_labels_pc = torch.mean(grouped_offsets, dim=2)

    return dir_labels_pc_cam, offset_labels_pc, grasp_success_labels_pc, approach_labels_pc_cam


def multi_bin_labels(cont_labels, bin_boundaries):
    """
    Computes binned grasp width labels from continous labels and bin boundaries

    Arguments:
        cont_labels {tf.Variable} -- continouos labels
        bin_boundaries {list} -- bin boundary values

    Returns:
        tf.Variable -- one/multi hot bin labels
    """
    bins = []
    for b in range(len(bin_boundaries) - 1):
        bins.append(
            torch.logical_and(torch.ge(cont_labels, bin_boundaries[b]), torch.lt(cont_labels, bin_boundaries[b + 1])))
    multi_hot_labels = torch.cat(bins, dim=2)
    multi_hot_labels = multi_hot_labels.type(torch.float32)

    return multi_hot_labels


def get_losses(pointclouds_pl, end_points, dir_labels_pc_cam, offset_labels_pc, grasp_success_labels_pc,
               approach_labels_pc_cam, global_config, device):
    """
    Computes loss terms from pointclouds, network predictions and labels

    Arguments:
        pointclouds_pl {tf.placeholder} -- bxNx3 input point clouds
        end_points {dict[str:tf.variable]} -- endpoints of the network containing predictions
        dir_labels_pc_cam {tf.variable} -- base direction labels in camera coordinates (bxNx3)
        offset_labels_pc {tf.variable} -- grasp width labels (bxNx1)
        grasp_success_labels_pc {tf.variable} -- contact success labels (bxNx1)
        approach_labels_pc_cam {tf.variable} -- approach direction labels in camera coordinates (bxNx3)
        global_config {dict} -- config dict

    Returns:
        [dir_cosine_loss, bin_ce_loss, offset_loss, approach_cosine_loss, adds_loss,
        adds_loss_gt2pred, gt_control_points, pred_control_points, pos_grasps_in_view] -- All losses (not all are used for training)
    """

    grasp_dir_head = end_points['grasp_dir_head']
    grasp_offset_head = end_points['grasp_offset_head']
    approach_dir_head = end_points['approach_dir_head']

    bin_weights = global_config['DATA']['labels']['bin_weights']
    tf_bin_weights = torch.tensor(bin_weights)

    min_geom_loss_divisor = torch.tensor(
        float(global_config['LOSS']['min_geom_loss_divisor'])) if 'min_geom_loss_divisor' in global_config[
        'LOSS'] else torch.tensor(1.)
    pos_grasps_in_view = torch.maximum(torch.sum(grasp_success_labels_pc, dim=1), min_geom_loss_divisor)

    ### ADS Gripper PC Loss
    if global_config['MODEL']['bin_offsets']:
        thickness_pred = get_bin_vals(global_config)[torch.argmax(grasp_offset_head, dim=2)]
        thickness_gt = get_bin_vals(global_config)[torch.argmax(offset_labels_pc, dim=2)]
    else:
        thickness_pred = grasp_offset_head[:, :, 0]
        thickness_gt = offset_labels_pc[:, :, 0]
    pred_grasps = build_6d_grasp(approach_dir_head, grasp_dir_head, pointclouds_pl,
                                 thickness_pred.to(device))  # b x num_point x 4 x 4
    gt_grasps_proj = build_6d_grasp(approach_labels_pc_cam, dir_labels_pc_cam, pointclouds_pl,
                                    thickness_gt.to(device))  # b x num_point x 4 x 4
    pos_gt_grasps_proj = torch.where(
        torch.broadcast_to(torch.unsqueeze(torch.unsqueeze(grasp_success_labels_pc.type(torch.bool), dim=2), dim=3),
                           gt_grasps_proj.shape), gt_grasps_proj, torch.ones_like(gt_grasps_proj) * 100000)
    # pos_gt_grasps_proj = tf.reshape(pos_gt_grasps_proj, (global_config['OPTIMIZER']['batch_size'], -1, 4, 4))

    gripper = mesh_utils.create_gripper('panda')
    gripper_control_points = gripper.get_control_point_tensor(global_config['OPTIMIZER']['batch_size'])  # b x 5 x 3
    sym_gripper_control_points = gripper.get_control_point_tensor(global_config['OPTIMIZER']['batch_size'],
                                                                  symmetric=True)

    gripper_control_points_homog = torch.cat([gripper_control_points, torch.ones(
        (global_config['OPTIMIZER']['batch_size'], gripper_control_points.shape[1], 1))], dim=2)  # b x 5 x 4
    sym_gripper_control_points_homog = torch.cat([sym_gripper_control_points, torch.ones(
        (global_config['OPTIMIZER']['batch_size'], gripper_control_points.shape[1], 1))], dim=2)  # b x 5 x 4

    # only use per point pred grasps but not per point gt grasps
    control_points = torch.unsqueeze(gripper_control_points_homog, dim=1).repeat(1, gt_grasps_proj.shape[1], 1,
                                                                                 1).to(device)  # b x num_point x 5 x 4
    sym_control_points = torch.unsqueeze(sym_gripper_control_points_homog, dim=1).repeat(1, gt_grasps_proj.shape[1], 1,
                                                                                         1).to(device)  # b x num_point x 5 x 4
    pred_control_points = torch.matmul(control_points, torch.transpose(pred_grasps, dim0=2, dim1=3))[:, :, :,
                          :3]  # b x num_point x 5 x 3

    ### Pred Grasp to GT Grasp ADD-S Loss
    gt_control_points = torch.matmul(control_points, torch.transpose(pos_gt_grasps_proj, dim0=2, dim1=3))[:, :, :,
                        :3].to(device)  # b x num_pos_grasp_point x 5 x 3
    sym_gt_control_points = torch.matmul(sym_control_points, torch.transpose(pos_gt_grasps_proj, dim0=2, dim1=3))[:, :,
                            :, :3].to(device) # b x num_pos_grasp_point x 5 x 3

    squared_add = torch.sum(
        (torch.unsqueeze(pred_control_points, dim=2) - torch.unsqueeze(gt_control_points, dim=1)) ** 2,
        dim=(3, 4))  # b x num_point x num_pos_grasp_point x ( 5 x 3)
    sym_squared_add = torch.sum(
        (torch.unsqueeze(pred_control_points, dim=2) - torch.unsqueeze(sym_gt_control_points, dim=1)) ** 2,
        dim=(3, 4))  # b x num_point x num_pos_grasp_point x ( 5 x 3)

    # symmetric ADD-S
    neg_squared_adds = -torch.cat([squared_add, sym_squared_add], dim=2)  # b x num_point x 2num_pos_grasp_point
    neg_squared_adds_k = torch.topk(neg_squared_adds, k=1, sorted=False)[0]  # b x num_point
    # If any pos grasp exists
    min_adds = torch.minimum(torch.sum(grasp_success_labels_pc, dim=1, keepdims=True),
                             torch.ones_like(neg_squared_adds_k[:, :, 0])) * torch.sqrt(-neg_squared_adds_k[:, :,
                                                                                         0])  # tf.minimum(tf.sqrt(-neg_squared_adds_k), tf.ones_like(neg_squared_adds_k)) # b x num_point
    adds_loss = torch.mean(end_points['binary_seg_pred'][:, :, 0] * min_adds)

    ### GT Grasp to pred Grasp ADD-S Loss
    gt_control_points = torch.matmul(control_points, torch.transpose(gt_grasps_proj, dim0=2, dim1=3))[:, :, :,
                        :3]  # b x num_pos_grasp_point x 5 x 3
    sym_gt_control_points = torch.matmul(sym_control_points, torch.transpose(gt_grasps_proj, dim0=2, dim1=3))[:, :, :,
                            :3]  # b x num_pos_grasp_point x 5 x 3

    neg_squared_adds = -torch.sum(
        (torch.unsqueeze(pred_control_points, dim=1) - torch.unsqueeze(gt_control_points, dim=2)) ** 2,
        dim=(3, 4))  # b x num_point x num_pos_grasp_point x ( 5 x 3)
    neg_squared_adds_sym = -torch.sum(
        (torch.unsqueeze(pred_control_points, dim=1) - torch.unsqueeze(sym_gt_control_points, dim=2)) ** 2,
        dim=(3, 4))  # b x num_point x num_pos_grasp_point x ( 5 x 3)

    neg_squared_adds_k_gt2pred, pred_grasp_idcs = torch.topk(neg_squared_adds, k=1,
                                                             sorted=False)  # b x num_pos_grasp_point
    neg_squared_adds_k_sym_gt2pred, pred_grasp_sym_idcs = torch.topk(neg_squared_adds_sym, k=1,
                                                                     sorted=False)  # b x num_pos_grasp_point
    pred_grasp_idcs_joined = torch.where(neg_squared_adds_k_gt2pred < neg_squared_adds_k_sym_gt2pred,
                                         pred_grasp_sym_idcs, pred_grasp_idcs)
    min_adds_gt2pred = torch.minimum(-neg_squared_adds_k_gt2pred,
                                     -neg_squared_adds_k_sym_gt2pred)  # b x num_pos_grasp_point x 1
    # min_adds_gt2pred = tf.math.exp(-min_adds_gt2pred)
    masked_min_adds_gt2pred = torch.multiply(min_adds_gt2pred[:, :, 0], grasp_success_labels_pc)

    batch_idcs = torch.meshgrid(torch.arange(pred_grasp_idcs_joined.shape[1]),
                                torch.arange(pred_grasp_idcs_joined.shape[0]), indexing="xy")
    gather_idcs = torch.stack((batch_idcs[1].to(device), pred_grasp_idcs_joined[:, :, 0]), dim=2)
    # change to here
    nearest_pred_grasp_confidence = end_points['binary_seg_pred'][:, :, 0][gather_idcs[:, :, 0], gather_idcs[:, :, 1]]
    adds_loss_gt2pred = torch.mean(
        torch.sum(nearest_pred_grasp_confidence * masked_min_adds_gt2pred, dim=1) / pos_grasps_in_view)

    ### Grasp baseline Loss
    cosine_distance = torch.tensor(1.).to(device) - torch.sum(torch.multiply(dir_labels_pc_cam, grasp_dir_head), dim=2)
    # only pass loss where we have labeled contacts near pc points
    masked_cosine_loss = torch.multiply(cosine_distance, grasp_success_labels_pc)
    dir_cosine_loss = torch.mean(torch.sum(masked_cosine_loss, dim=1) / pos_grasps_in_view)

    ### Grasp Approach Loss
    approach_labels_orthog = F.normalize(
        approach_labels_pc_cam - torch.sum(torch.multiply(grasp_dir_head, approach_labels_pc_cam), dim=2,
                                           keepdims=True) * grasp_dir_head, dim=2)
    cosine_distance_approach = torch.tensor(1.).to(device) - torch.sum(torch.multiply(approach_labels_orthog, approach_dir_head),
                                                            dim=2)
    masked_approach_loss = torch.multiply(cosine_distance_approach, grasp_success_labels_pc)
    approach_cosine_loss = torch.mean(torch.sum(masked_approach_loss, dim=1) / pos_grasps_in_view)

    ### Grasp Offset/Thickness Loss
    if global_config['MODEL']['bin_offsets']:
        if global_config['LOSS']['offset_loss_type'] == 'softmax_cross_entropy':
            # offset_loss_old = tf.losses.softmax_cross_entropy(tf.constant(offset_labels_pc.detach().numpy()), tf.constant(grasp_offset_head.detach().numpy()))
            offset_loss = torch.zeros(grasp_offset_head.shape[0], grasp_offset_head.shape[1])
            for batch in range(offset_loss.shape[0]):
                offset_loss[batch] = F.cross_entropy(grasp_offset_head[batch],
                                                     torch.argmax(offset_labels_pc[batch], dim=1), reduction='none')
            offset_loss = torch.mean(offset_loss)
        else:
            offset_loss = offset_labels_pc * -torch.log(torch.sigmoid(grasp_offset_head)) + (
                        1 - offset_labels_pc) * -torch.log(1 - torch.sigmoid(grasp_offset_head))
            # offset_loss_old = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(offset_labels_pc.detach().numpy()), logits=tf.constant(grasp_offset_head.detach().numpy()))

            if 'too_small_offset_pred_bin_factor' in global_config['LOSS'] and global_config['LOSS'][
                'too_small_offset_pred_bin_factor']:
                too_small_offset_pred_bin_factor = torch.tensor(
                    global_config['LOSS']['too_small_offset_pred_bin_factor'], torch.float32)
                # collision_weight = tf.math.cumsum(offset_labels_pc, axis=2,
                #                                   reverse=True) * too_small_offset_pred_bin_factor + torch.constant(1.)
                collision_weight = (offset_labels_pc + torch.sum(offset_labels_pc, dim=2, keepdims=True) - torch.cumsum(
                    offset_labels_pc, dim=2)) \
                                   * too_small_offset_pred_bin_factor + torch.constant(1.)
                offset_loss = torch.multiply(collision_weight, offset_loss)

            offset_loss = torch.mean(torch.multiply(torch.reshape(tf_bin_weights.to(device), (1, 1, -1)), offset_loss), axis=2)
    else:
        offset_loss = (grasp_offset_head[:, :, 0] - offset_labels_pc[:, :, 0]) ** 2
    masked_offset_loss = torch.multiply(offset_loss, grasp_success_labels_pc)
    offset_loss = torch.mean(torch.sum(masked_offset_loss, dim=1) / pos_grasps_in_view)

    ### Grasp Confidence Loss
    # bin_ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(grasp_success_labels_pc, axis=2),
    #                                                       logits=end_points['binary_seg_head'])
    bin_ce_loss = torch.unsqueeze(grasp_success_labels_pc, dim=2) * -torch.log(
        torch.sigmoid(end_points['binary_seg_head'])) + (
                              1 - torch.unsqueeze(grasp_success_labels_pc, dim=2)) * -torch.log(
        1 - torch.sigmoid(end_points['binary_seg_head']))
    if 'topk_confidence' in global_config['LOSS'] and global_config['LOSS']['topk_confidence']:
        bin_ce_loss, _ = torch.topk(torch.squeeze(bin_ce_loss), k=global_config['LOSS']['topk_confidence'])
    bin_ce_loss = torch.mean(bin_ce_loss)

    return dir_cosine_loss, bin_ce_loss, offset_loss, approach_cosine_loss, adds_loss, adds_loss_gt2pred


