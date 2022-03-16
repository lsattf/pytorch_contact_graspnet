import torch
from predict_dataset import ContactDataset, show_image, visualize_grasps
from network import ContactGraspNet, get_grasp_and_score
import numpy as np

model_path = 'finished_model'
input_paths = 'test_data/0.npy'

end_point_test = np.load('end_point_test.npy', allow_pickle='TRUE').item()
pc_mean = np.load('pc_mean.npy', allow_pickle='TRUE')
pc_batch = np.load('pc_batch.npy', allow_pickle='TRUE')
dataset = ContactDataset(input_paths=input_paths)

test = ContactGraspNet(model_path)
test.load_state_dict(torch.load('checkpoints/ContactGraspNet.pt')['model_state_dict'])
test.eval()
saved_pc_batch_pred = test(torch.transpose(torch.tensor(pc_batch), dim0=1, dim1=2))
raw_pred = test(torch.transpose(torch.tensor(dataset[2]), dim0=1, dim1=2))

show_image(dataset.getrgb()[0], dataset.getsegmap()[0])

# predict grasp with original code
end_point_test_grasp, end_point_test_score = get_grasp_and_score(test.getcfg(), end_point_test[2], torch.tensor(pc_mean), dataset.getpc_segments(), filter_grasp=dataset.filter_grasps, k=2, use_end_point_test=True)
print('Generated {} grasps'.format(len(end_point_test_grasp)))
visualize_grasps(dataset.getpc_full()[0], {2: end_point_test_grasp.detach().numpy()}, {2: end_point_test_score.detach().numpy()}, plot_opencv_cam=True, pc_colors=dataset.getpc_colors()[0])

# predict grasp with current network but original data preprocessing
end_point_test_grasp, end_point_test_score = get_grasp_and_score(test.getcfg(), saved_pc_batch_pred, torch.tensor(pc_mean), dataset.getpc_segments(), filter_grasp=dataset.filter_grasps, k=2)
print('Generated {} grasps'.format(len(end_point_test_grasp)))
visualize_grasps(dataset.getpc_full()[0], {2: end_point_test_grasp.detach().numpy()}, {2: end_point_test_score.detach().numpy()}, plot_opencv_cam=True, pc_colors=dataset.getpc_colors()[0])

# predict grasp with current network and current data preprocessing
raw_pred_grasp, raw_pred_score = get_grasp_and_score(test.getcfg(), raw_pred, torch.tensor(dataset.getpc_mean()[0][2]), dataset.getpc_segments(), filter_grasp=dataset.filter_grasps, k=2)
print('Generated {} grasps'.format(len(raw_pred_grasp)))
visualize_grasps(dataset.getpc_full()[0], {2: raw_pred_grasp.detach().numpy()}, {2: raw_pred_score.detach().numpy()}, plot_opencv_cam=True, pc_colors=dataset.getpc_colors()[0])




