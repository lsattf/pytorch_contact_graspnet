import os
from get_labels_and_losses import *
from network import load_config, ContactGraspNet
from data import PointCloudReader, center_pc_convert_cam, load_scene_contacts
import torch


model_path = 'checkpoints'
data_path = 'acronym'
global_config = load_config(model_path, save=False)
contact_infos = load_scene_contacts(global_config['DATA']['data_path'], scene_contacts_path=global_config['DATA']['scene_contacts_path'])

num_train_samples = len(contact_infos) - global_config['DATA']['num_test_scenes']
num_test_samples = global_config['DATA']['num_test_scenes']

print('using %s meshes' % (num_train_samples + num_test_samples))
if 'train_and_test' in global_config['DATA'] and global_config['DATA']['train_and_test']:
    num_train_samples = num_train_samples + num_test_samples
    num_test_samples = 0
    print('using train and test data')

# real data compare to y_data
pcreader = PointCloudReader(
    root_folder=global_config['DATA']['data_path'],
    batch_size=global_config['OPTIMIZER']['batch_size'],
    estimate_normals=global_config['DATA']['input_normals'],
    raw_num_points=global_config['DATA']['raw_num_points'],
    use_uniform_quaternions=global_config['DATA']['use_uniform_quaternions'],
    scene_obj_scales=[c['obj_scales'] for c in contact_infos],
    scene_obj_paths=[c['obj_paths'] for c in contact_infos],
    scene_obj_transforms=[c['obj_transforms'] for c in contact_infos],
    num_train_samples=num_train_samples,
    num_test_samples=num_test_samples,
    use_farthest_point=global_config['DATA']['use_farthest_point'],
    intrinsics=global_config['DATA']['intrinsics'],
    elevation=global_config['DATA']['view_sphere']['elevation'],
    distance_range=global_config['DATA']['view_sphere']['distance_range'],
    pc_augm_config=global_config['DATA']['pc_augm'],
    depth_augm_config=global_config['DATA']['depth_augm']
)

max_epoch = global_config['OPTIMIZER']['max_epoch']
torch.cuda.empty_cache()
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")
model = ContactGraspNet(model_path).to(device)
pos_contact_points, pos_contact_dirs, pos_contact_approaches, pos_finger_diffs, scene_idcs = load_contact_grasps(contact_infos, global_config['DATA'])

# print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=global_config['OPTIMIZER']['learning_rate'])
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                      gamma=global_config['OPTIMIZER']['decay_rate'])

current_epoch = 0
if os.path.exists('checkpoints/ContactGraspNet.pt'):
    checkpoint = torch.load('checkpoints/ContactGraspNet.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    current_epoch = checkpoint['epoch']

model.train()

for epoch in range(current_epoch, max_epoch + 1):
    print(f"Epoch {epoch}\n-------------------------------")
    loss_for_this_epoch = 0
    for batch_idx in range(pcreader._num_train_samples):
        batch_data, cam_poses, scene_idx = pcreader.get_scene_batch(scene_idx=batch_idx)

        # OpenCV OpenGL conversion
        cam_poses, batch_data = center_pc_convert_cam(cam_poses, batch_data)
        end_points = model(torch.transpose(torch.tensor(batch_data, device=device), dim0=1, dim1=2))
        # end_points = model(torch.transpose(torch.tensor(batch_data), dim0=1, dim1=2))

        # Get labels
        dir_labels_pc_cam, offset_labels_pc, grasp_suc_labels_pc, approach_labels_pc = compute_labels(
            pos_contact_points[batch_idx].to(device), pos_contact_dirs[batch_idx].to(device), pos_contact_approaches[batch_idx].to(device),
            pos_finger_diffs[batch_idx].to(device), end_points['pred_points'], cam_poses, global_config, device=device)

        if global_config['MODEL']['bin_offsets']:
            orig_offset_labels = offset_labels_pc
            offset_labels_pc = torch.abs(offset_labels_pc)
            offset_labels_pc = multi_bin_labels(offset_labels_pc, global_config['DATA']['labels']['offset_bins'])

        # Get losses
        dir_loss, bin_ce_loss, offset_loss, approach_loss, adds_loss, adds_loss_gt2pred = get_losses(
            end_points['pred_points'], end_points, dir_labels_pc_cam, offset_labels_pc, grasp_suc_labels_pc,
            approach_labels_pc, global_config, device=device)

        total_loss = 0
        if global_config['MODEL']['pred_contact_base']:
            total_loss += global_config['OPTIMIZER']['dir_cosine_loss_weight'] * dir_loss
        if global_config['MODEL']['pred_contact_success']:
            total_loss += global_config['OPTIMIZER']['score_ce_loss_weight'] * bin_ce_loss
        if global_config['MODEL']['pred_contact_offset']:
            total_loss += global_config['OPTIMIZER']['offset_loss_weight'] * offset_loss
        if global_config['MODEL']['pred_contact_approach']:
            total_loss += global_config['OPTIMIZER']['approach_cosine_loss_weight'] * approach_loss
        if global_config['MODEL']['pred_grasps_adds']:
            total_loss += global_config['OPTIMIZER']['adds_loss_weight'] * adds_loss
        if global_config['MODEL']['pred_grasps_adds_gt2pred']:
            total_loss += global_config['OPTIMIZER']['adds_gt2pred_loss_weight'] * adds_loss_gt2pred

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        loss_for_this_epoch = total_loss
        if batch_idx % 1 == 0:
            print(f"loss: {total_loss:>7f}  at batch: {batch_idx:>5d}/{num_train_samples:>5d}]")
        if batch_idx % 200 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_for_this_epoch}, 'checkpoints/ContactGraspNet.pt')

    if (epoch + 1) % 10 == 0:
        lr_scheduler.step()
        if optimizer.param_groups[0]['lr'] <= 0.00001:
            optimizer.param_groups[0]['lr'] = 0.00001

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_for_this_epoch}, 'checkpoints/ContactGraspNet.pt')
