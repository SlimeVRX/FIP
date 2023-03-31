r"""
    Test the system with an example IMU measurement sequence.
"""


import torch
from net import PoseNet
from config import paths, joint_set
from utils import normalize_and_concat
import os
import articulate as art
sample_idx = 2

def Our():
    isMatrix = False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = PoseNet(isMatrix=isMatrix, device=device).to(device)
    net.load_state_dict(torch.load("weights.tar"))
    net.eval()
    data = torch.load(os.path.join(paths.dipimu_dir, 'test.pt'))
    acc = data['acc'][sample_idx]
    ori = data['ori'][sample_idx]
    # print(acc[0].shape, ori[0].shape)
    # acc = torch.load(os.path.join(paths.example_dir, 'acc.pt'))
    # ori = torch.load(os.path.join(paths.example_dir, 'ori.pt'))
    # print(acc.shape, ori.shape)

    x = normalize_and_concat(acc, ori,isMatrix=isMatrix).to(device)
    x = x.unsqueeze(1)
    pose, tran, contact_probability = net.forward_offline(x)     # offline
    # pose, tran = [torch.stack(_) for _ in zip(*[net.forward_online(f) for f in x])]   # online
    # tran = torch.zeros((len(pose), 3)).to(device)
    art.ParametricModel(paths.male_smpl_file, device=device).view_motion([pose], [tran], contact=contact_probability)

# clone Transpose git

# def tranPose():
#     # Transose
#     isMatrix = True
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     net = TransPoseNet().to(device)
#     checkpoint = torch.load("data/weights.pt")
#     net.load_state_dict(checkpoint)
#     net.eval()
#     data = torch.load(os.path.join(paths.dipimu_dir, 'test.pt'))
#     acc = data['acc'][sample_idx]
#     ori = data['ori'][sample_idx]
#     x = normalize_and_concat(acc, ori,isMatrix=isMatrix).to(device)
#     x = x.unsqueeze(1)
#     # pose, tran = net.forward_offline(x)     # offline
    
#     # pose = pose.cuda()
#     pose, tran = [torch.stack(_) for _ in zip(*[net.forward_online(f) for f in x])]   # online
#     tran = torch.zeros((len(pose), 3)).to(device)
#     art.ParametricModel(paths.male_smpl_file, device=device).view_motion([pose], [tran])

def DIP():
    # DIP

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = torch.load(os.path.join(paths.dipimu_dir, 'test.pt'))
    pose = data['pose'][sample_idx]
    root_rotation = data['ori'][sample_idx][:, -1]


    pose = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3).to(device)


    m = art.ParametricModel(paths.male_smpl_file, device=device)
    global_to_local_pose = m.inverse_kinematics_R
    local_to_global_pose = m.forward_kinematics_R
    global_pose = local_to_global_pose(pose)
    glb_reduced_pose = global_pose[:, joint_set.reduced]

    def _reduced_glb_6d_to_full_local_mat(root_rotation, glb_reduced_pose):
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:,joint_set.reduced] = glb_reduced_pose
        pose = global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = root_rotation.view(-1, 3, 3)
        return pose
    pose = _reduced_glb_6d_to_full_local_mat(root_rotation, glb_reduced_pose)
    print(pose.shape)
    tran = torch.zeros((len(pose), 3)).to(device)
    # pose, tran = [torch.stack(_) for _ in zip(*[net.forward_online(f) for f in x])]   # online
    art.ParametricModel(paths.male_smpl_file, device=device).view_motion([pose], [tran])

# tranPose()
Our()