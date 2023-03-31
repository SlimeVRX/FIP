r"""
    Utils for the project.
"""


import torch
from config import acc_scale
import articulate as art
import yaml


def normalize_and_concat(glb_acc, glb_ori, num_joints_in=6, isMatrix=True, no_norm=False, return_root=False, onlyori=False):
    glb_acc = glb_acc.view(-1, num_joints_in, 3)
    glb_ori = glb_ori.view(-1, num_joints_in, 3, 3)
    if not no_norm:
        acc = torch.cat((glb_acc[:, :(num_joints_in-1)] - glb_acc[:, (num_joints_in-1):], glb_acc[:, (num_joints_in-1):]), dim=1).bmm(glb_ori[:, -1]) / acc_scale
        ori = torch.cat((glb_ori[:, (num_joints_in-1):].transpose(2, 3).matmul(glb_ori[:, :(num_joints_in-1)]), glb_ori[:, (num_joints_in-1):]), dim=1)
        if onlyori:
            if not isMatrix:
                ori = art.math.rotation_matrix_to_r6d(ori).view(-1, num_joints_in, 6)
            return ori.flatten(1)
    else:
        acc = (glb_acc[:, :(num_joints_in-1)] - glb_acc[:, (num_joints_in-1):]).bmm(glb_ori[:, -1]) / acc_scale
        ori = glb_ori[:, (num_joints_in-1):].transpose(2, 3).matmul(glb_ori[:, :(num_joints_in-1)])
        
    if not isMatrix:
        if no_norm:
            num_joints_in -= 1
        ori = art.math.rotation_matrix_to_r6d(ori).view(-1, num_joints_in, 6)
    data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=-1)
    if return_root:
        return data, glb_ori[:, -1]
    return data


def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    file.close()
    return dict
from config import joint_set, paths

def reduced_glb_to_full_local_mat(root_rotation, glb_reduced_pose):
    body_model = art.model.ParametricModel(paths.male_smpl_file)
    global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
    global_full_pose[:, joint_set.reduced] = glb_reduced_pose
    pose = body_model.inverse_kinematics_R(global_full_pose).view(-1, 24, 3, 3)
    pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
    pose[:, 0] = root_rotation.view(-1, 3, 3)
    return pose

class PoseFilter:
    def __init__(self, alpha=0.6):
        self.last_quat = None
        self.alpha = alpha
    def reset(self):
        self.last_quat = None
    def update(self, new_quat:torch.Tensor):
        """[过滤函数]

        Args:
            new_ori (torch.Tensor): [24, 4]
        """
        if self.last_quat is None:
            self.last_quat = new_quat
        normal_quat = self.filter_deabnormal(new_quat, self.last_quat)
        lp_quat = self.filter_lowpass(normal_quat, self.last_quat)
        self.last_quat = lp_quat
        return self.last_quat
    
    def filter_deabnormal(self, data_new:torch.Tensor, data_last:torch.Tensor):
        ''' deabnormal filter, remove the abnormal value.

            Parameter:
            data_new : the new IMU data
            data_last: the last IMU data

            Return:
            data_normal: the IMU data that have remove abnormal value
        '''
        data_normal = data_new
        indice = torch.abs(data_new - data_last) > 1
        data_normal[indice] = data_last[indice]

        return data_normal

    def filter_lowpass(self, data_new:torch.Tensor, data_last:torch.Tensor):
        ''' low pass filter.

            Parameter:
            data_new : the new IMU data
            data_last: the last IMU data

            Return:
            data_lowpass: the low frequencey IMU data
        '''
        data_lowpass = data_last*self.alpha + data_new*(1-self.alpha)
        return data_lowpass
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    x = torch.randn(300, 24, 4)
    poseFilter = PoseFilter()
    filtered_data=[]
    for i in x:
        filtered_data.append(poseFilter.update(i))
    filtered_data = torch.stack(filtered_data)

    x_axis = np.linspace(0,2,len(x))
    fig,ax = plt.subplots()
    ax.plot(x_axis,x[:, 0, 0],label='origin')
    ax.legend()
    plt.savefig("a.png")

    x_axis = np.linspace(0,2,len(x))
    fig,ax = plt.subplots()
    ax.plot(x_axis,filtered_data[:, 0, 0],label='filtered')
    ax.legend()
    plt.savefig("b.png")
