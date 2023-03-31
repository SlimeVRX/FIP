from torch.utils.data import DataLoader, Dataset
import torch
import articulate
from config import paths, joint_set
import config
from utils import normalize_and_concat
import os

# class OwnDatasets(Dataset):
#     def __init__(self, filepath, use_joint=[0, 1, 2, 3, 4, 5]):
#         super(OwnDatasets, self).__init__()
#         data = torch.load(filepath)
#         self.use_joint = use_joint
#         self.pose = data['pose']
#         self.tran = data['tran']
#         self.ori = data['ori']
#         self.acc = data['acc']
#         self.point = data['jp']

#     def __getitem__(self, idx):
#         nn_pose = self.pose[idx].float()
#         if self.tran[idx] is not None:
#             tran = self.tran[idx].float()
#         ori = self.ori[idx][:, self.use_joint].float()
#         acc = self.acc[idx][:, self.use_joint].float()
#         joint_pos = self.point[idx].float()
#         root_ori = ori[:, -1] # 最后一组为胯部
#         imu = normalize_and_concat(acc, ori)

#         # 世界速度->本地速度
#         if self.tran[idx] is not None:
#             velocity = tran
#             velocity_local = root_ori.transpose(1, 2).bmm(
#                 torch.cat((torch.zeros(1, 3), velocity[1:] - velocity[:-1])).unsqueeze(-1)).squeeze(-1) * 60 / config.vel_scale
#         else:
#             velocity_local = torch.zeros((len(imu), 3))
#         # 支撑腿
#         stable_threshold = 0.008
#         diff = joint_pos - torch.cat((joint_pos[:1], joint_pos[:-1]))
#         stable = (diff[:, [7, 8]].norm(dim=2) < stable_threshold).float()

#         # 关节位置
#         nn_jtr = joint_pos - joint_pos[:, :1]
        
#         leaf_jtr = nn_jtr[:, joint_set.leaf]
#         full_jtr = nn_jtr[:, joint_set.full]

#         return imu, nn_pose.flatten(1),leaf_jtr.flatten(1), full_jtr.flatten(1), stable, velocity_local, root_ori

#     def __len__(self):
#         return len(self.ori)


class OwnDatasets(Dataset):
    def __init__(self, filepath, use_joint=[0, 1, 2, 3, 4, 5], isMatrix=True, no_norm=False, onlyori=False):
        super(OwnDatasets, self).__init__()
        data = torch.load(filepath)
        self.use_joint = use_joint
        self.pose = data['pose']
        self.tran = data['tran']
        self.ori = data['ori']
        self.acc = data['acc']
        self.point = data['jp']
        self.isMatrix = isMatrix
        self.no_norm = no_norm
        self.onlyori = onlyori

        self.m = articulate.ParametricModel(paths.male_smpl_file)
        self.global_to_local_pose = self.m.inverse_kinematics_R


    def __getitem__(self, idx):
        nn_pose = self.pose[idx].float()
        if self.tran[idx] is not None:
            tran = self.tran[idx].float()
        ori = self.ori[idx][:, self.use_joint].float()
        acc = self.acc[idx][:, self.use_joint].float()
        joint_pos = self.point[idx].float()
        root_ori = ori[:, -1] # 最后一组为胯部
        imu = normalize_and_concat(acc, ori, len(self.use_joint), self.isMatrix, self.no_norm, onlyori=self.onlyori )

        # 世界速度->本地速度
        if self.tran[idx] is not None:
            velocity = tran
            velocity_local = root_ori.transpose(1, 2).bmm(
                torch.cat((torch.zeros(1, 3), velocity[1:] - velocity[:-1])).unsqueeze(-1)).squeeze(-1) * 60 / config.vel_scale
        else:
            velocity_local = torch.zeros((len(nn_pose), 3))
        # 支撑腿
        stable_threshold = 0.008
        diff = joint_pos - torch.cat((joint_pos[:1], joint_pos[:-1]))
        stable = (diff[:, [7, 8]].norm(dim=2) < stable_threshold).float()

        # 关节位置
        # nn_jtr = joint_pos - joint_pos[:, :1]
        
        # leaf_jtr = nn_jtr[:, joint_set.leaf]
        # full_jtr = nn_jtr[:, joint_set.full]
        full_pose = self._reduced_glb_6d_to_full_local_mat(root_ori, nn_pose)
        pose_global, joint_global = self.m.forward_kinematics(full_pose)
        nn_jtr = joint_global - joint_global[:, :1]
        leaf_jtr = nn_jtr[:, joint_set.leaf]
        full_jtr = nn_jtr[:, joint_set.full]

        return imu, nn_pose.flatten(1),leaf_jtr.flatten(1), full_jtr.flatten(1), stable, velocity_local, root_ori

    def __len__(self):
        return len(self.ori)

    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = articulate.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = root_rotation.view(-1, 3, 3)
        return pose

if __name__ == "__main__":
    dataset = OwnDatasets(os.path.join(paths.dipimu_dir, "veri.pt"))
    for imu, nn_pose,leaf_jtr, full_jtr, stable, velocity_local, root_ori in dataset:
        print(imu.shape)
        print(nn_pose.shape)
        print(leaf_jtr.shape)
        print(full_jtr.shape)
        print(stable.shape)
        if velocity_local is not None:
            print(velocity_local.shape)
        print(root_ori.shape)
        break