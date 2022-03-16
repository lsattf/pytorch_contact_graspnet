import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2.models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
from get_grasp import *


class ContactGraspNet(nn.Module):
    def __init__(self, model_path, is_training=False, bn_decay=None):
        super(ContactGraspNet, self).__init__()
        self._cfg = load_config(model_path, batch_size=1)
        self._is_training = is_training
        self._bn_decay = bn_decay
        self._model_config = self._cfg['MODEL']
        self._data_config = self._cfg['DATA']
        self._end_points = {}

        self._radius_list_0 = self._model_config['pointnet_sa_modules_msg'][0]['radius_list']
        self._radius_list_1 = self._model_config['pointnet_sa_modules_msg'][1]['radius_list']
        self._radius_list_2 = self._model_config['pointnet_sa_modules_msg'][2]['radius_list']

        self._nsample_list_0 = self._model_config['pointnet_sa_modules_msg'][0]['nsample_list']
        self._nsample_list_1 = self._model_config['pointnet_sa_modules_msg'][1]['nsample_list']
        self._nsample_list_2 = self._model_config['pointnet_sa_modules_msg'][2]['nsample_list']

        self._mlp_list_0 = self._model_config['pointnet_sa_modules_msg'][0]['mlp_list']
        self._mlp_list_1 = self._model_config['pointnet_sa_modules_msg'][1]['mlp_list']
        self._mlp_list_2 = self._model_config['pointnet_sa_modules_msg'][2]['mlp_list']

        self._npoint_0 = self._model_config['pointnet_sa_modules_msg'][0]['npoint']
        self._npoint_1 = self._model_config['pointnet_sa_modules_msg'][1]['npoint']
        self._npoint_2 = self._model_config['pointnet_sa_modules_msg'][2]['npoint']

        self._fp_mlp_0 = self._model_config['pointnet_fp_modules'][0]['mlp']
        self._fp_mlp_1 = self._model_config['pointnet_fp_modules'][1]['mlp']
        self._fp_mlp_2 = self._model_config['pointnet_fp_modules'][2]['mlp']

        self._input_normals = self._data_config['input_normals']
        self._offset_bins = self._data_config['labels']['offset_bins']
        self._joint_heads = self._model_config['joint_heads']
        self.dp_70 = torch.nn.Dropout(p=0.7)
        self.dp_50 = torch.nn.Dropout(p=0.5)
        self.layer1 = PointNetSetAbstractionMsg(self._npoint_0, self._radius_list_0, self._nsample_list_0, 0, self._mlp_list_0)
        self.layer2 = PointNetSetAbstractionMsg(self._npoint_1, self._radius_list_1, self._nsample_list_1, 320, self._mlp_list_1)#changed
        if 'asymmetric_model' in self._model_config and self._model_config['asymmetric_model']:
            self.layer3 = PointNetSetAbstractionMsg(self._npoint_2, self._radius_list_2, self._nsample_list_2, 640, self._mlp_list_2)#changed
        else:
            self.layer3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=640, mlp=self._model_config['pointnet_sa_module']['mlp'], group_all=self._model_config['pointnet_sa_module']['group_all'])#changed
        self.layer4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=643, mlp=self._model_config['pointnet_sa_module']['mlp'], group_all=self._model_config['pointnet_sa_module']['group_all'])
        self.fa_layer1 = PointNetFeaturePropagation(1664, self._fp_mlp_0)
        self.fa_layer2 = PointNetFeaturePropagation(896, self._fp_mlp_1)
        self.fa_layer3 = PointNetFeaturePropagation(448, self._fp_mlp_2)
        self.fc1 = torch.nn.Conv1d(128, 128, 1)
        self.fc1_batchnorm1d = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Conv1d(128, 4, 1)
        self.fc3 = torch.nn.Conv1d(128, 3, 1)
        self.fc1_app = torch.nn.Conv1d(128, 128, 1)
        self.fc1_app_batchnorm1d = torch.nn.BatchNorm1d(128)
        self.fc3_app = torch.nn.Conv1d(128, 3, 1)
        self.fc1_off = torch.nn.Conv1d(128, 128, 1)
        self.fc1_off_batchnorm1d = torch.nn.BatchNorm1d(128)
        self.fc2_off = torch.nn.Conv1d(128, len(self._offset_bins) - 1, 1)
        self.fc1_seg = torch.nn.Conv1d(128, 128, 1)
        self.fc1_seg_batchnorm1d = torch.nn.BatchNorm1d(128)
        self.fc2_seg = torch.nn.Conv1d(128, 1, 1)

    def forward(self, point_cloud):
        # if 'raw_num_points' in data_config and data_config['raw_num_points'] != data_config['ndataset_points']:
        #     point_cloud = gather_point(point_cloud, farthest_point_sample(data_config['ndataset_points'], point_cloud))------skip
        end_points = {}

        l0_xyz = point_cloud[0:, 0:3, 0:]
        l0_points = point_cloud[0:, 3:6, 0:] if self._input_normals else None
        l1_xyz, l1_points = self.layer1(l0_xyz, l0_points)

        l2_xyz, l2_points = self.layer2(l1_xyz, l1_points)
        self._model_config['asymmetric_model'] = False
        if 'asymmetric_model' in self._model_config and self._model_config['asymmetric_model']:
            l3_xyz, l3_points = self.layer3(l2_xyz, l2_points)
            l4_xyz, l4_points = self.layer4(l3_xyz, l3_points)#changed

            # # Feature Propagation layers
            l3_points = self.fa_layer1(l3_xyz, l4_xyz, l3_points, l4_points)
            l2_points = self.fa_layer2(l2_xyz, l3_xyz, l2_points, l3_points)
            l1_points = self.fa_layer3(l1_xyz, l2_xyz, l1_points, l2_points)

            l0_points = l1_points
            pred_points = l1_xyz
        else:
            l3_xyz, l3_points = self.layer3(l2_xyz, l2_points) #changed
            #redefine feature propagation layers
            self.fa_layer1 = PointNetFeaturePropagation(1280, self._fp_mlp_0)
            self.fa_layer2 = PointNetFeaturePropagation(576, self._fp_mlp_1)

            # Feature Propagation layers
            l2_points = self.fa_layer1(l2_xyz, l3_xyz, l2_points, l3_points)
            l1_points = self.fa_layer2(l1_xyz, l2_xyz, l1_points, l2_points)
            l0_points = torch.cat((l0_xyz, l0_points), dim=-2) if self._input_normals else l0_xyz
            self.fa_layer3 = PointNetFeaturePropagation(l0_points.shape[1]+l1_points.shape[1], self._fp_mlp_2)
            l0_points = self.fa_layer3(l0_xyz, l1_xyz, l0_points, l1_points)
            pred_points = l0_xyz
        #
        if self._joint_heads:
            head = self.fc1(l0_points)
            head = F.relu(self.fc1_batchnorm1d(head))
            if self._is_training:
                head = self.dp_70(head)
            head = self.fc2(head)
            grasp_dir_head = head[0:, 0:, 0:3]
            grasp_dir_head_normed = F.normalize(grasp_dir_head, p=2, dim=1)
            binary_seg_head = head[0:, 0:, 3:4]
        else:
            # Head for grasp direction
            grasp_dir_head = self.fc1(l0_points)
            grasp_dir_head = F.relu(self.fc1_batchnorm1d(grasp_dir_head))
            if self._is_training:
                grasp_dir_head = self.dp_70(grasp_dir_head)
            grasp_dir_head = self.fc3(grasp_dir_head)
            grasp_dir_head_normed = F.normalize(grasp_dir_head, p=2, dim=1)

            # Head for grasp approach
            approach_dir_head = self.fc1_app(l0_points)
            approach_dir_head = F.relu(self.fc1_app_batchnorm1d(approach_dir_head))
            if self._is_training:
                approach_dir_head = self.dp_70(approach_dir_head)
            approach_dir_head = self.fc3_app(approach_dir_head)
            approach_dir_head_orthog = F.normalize(approach_dir_head - torch.sum(torch.mul(grasp_dir_head_normed, approach_dir_head), 1, keepdims=True) * grasp_dir_head_normed, p=2, dim=1)

            # Head for grasp width
            if self._model_config['dir_vec_length_offset']:
                grasp_offset_head = torch.norm(grasp_dir_head, dim=1, keepdims=True)
            elif self._model_config['bin_offsets']:
                grasp_offset_head = self.fc1_off(l0_points)
                grasp_offset_head = F.relu(self.fc1_off_batchnorm1d(grasp_offset_head))
                grasp_offset_head = self.fc2_off(grasp_offset_head)
            else:
                grasp_offset_head = self.fc1_off(l0_points)
                grasp_offset_head = F.relu(self.fc1_off_batchnorm1d(grasp_offset_head))
                if self._is_training:
                    grasp_offset_head = self.dp_70(grasp_offset_head)
                grasp_offset_head = self.fc2_off(grasp_offset_head)

            # Head for contact points
            binary_seg_head = self.fc1_seg(l0_points)
            binary_seg_head = F.relu(self.fc1_seg_batchnorm1d(binary_seg_head))
            if self._is_training:
                binary_seg_head = self.dp_50(binary_seg_head)
            binary_seg_head = self.fc2_seg(binary_seg_head)

        end_points['grasp_dir_head'] = grasp_dir_head_normed
        end_points['binary_seg_head'] = binary_seg_head
        end_points['binary_seg_pred'] = torch.sigmoid(binary_seg_head) #changed
        end_points['grasp_offset_head'] = grasp_offset_head
        end_points['grasp_offset_pred'] = torch.sigmoid(grasp_offset_head) if self._model_config['bin_offsets'] else grasp_offset_head #changed
        end_points['approach_dir_head'] = approach_dir_head_orthog
        end_points['pred_points'] = pred_points

        # swap the axis of result matrices
        end_points['grasp_dir_head'] = torch.transpose(end_points['grasp_dir_head'], dim0=1, dim1=2)
        end_points['binary_seg_head'] = torch.transpose(end_points['binary_seg_head'], dim0=1, dim1=2)
        end_points['binary_seg_pred'] = torch.transpose(end_points['binary_seg_pred'], dim0=1, dim1=2)
        end_points['grasp_offset_head'] = torch.transpose(end_points['grasp_offset_head'], dim0=1, dim1=2)
        end_points['grasp_offset_pred'] = torch.transpose(end_points['grasp_offset_pred'], dim0=1, dim1=2)
        end_points['approach_dir_head'] = torch.transpose(end_points['approach_dir_head'], dim0=1, dim1=2)
        end_points['pred_points'] = torch.transpose(end_points['pred_points'], dim0=1, dim1=2)

        self._end_points = end_points

        return end_points

    def getcfg(self):
        return self._cfg






