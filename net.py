import torch.nn
from torch.nn.functional import relu
from config import *
import articulate as art
from sru import SRU
from utils import PoseFilter

class RNN_SRU(torch.nn.Module):
    r"""
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=5, bidirectional=True, dropout=0.2):
        super(RNN_SRU, self).__init__()
        self.rnn = SRU(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional, layer_norm=True, dropout=dropout)
        self.linear1 = torch.nn.Linear(n_input, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, h=None):
        x, h = self.rnn(self.dropout(relu(self.linear1(x))), h)
        return self.linear2(x), h

class PoseNet(torch.nn.Module):
    r"""
    Whole pipeline for pose and translation estimation.
    """
    def __init__(self, num_past_frame=20, num_future_frame=5, hip_length=None, upper_leg_length=None,
                 lower_leg_length=None, prob_threshold=(0.5, 0.9), gravity_velocity=-0.018, isMatrix=True, num_joint=6, onlyori=False, device=torch.device("cpu"), smooth_alpha=0.6):
        r"""
        :param num_past_frame: Number of past frames for a biRNN window.
        :param num_future_frame: Number of future frames for a biRNN window.
        :param hip_length: Hip length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param upper_leg_length: Upper leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param lower_leg_length: Lower leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param prob_threshold: The probability threshold used to control the fusion of the two translation branches.
        :param gravity_velocity: The gravity velocity added to the Trans-B1 when the body is not on the ground.
        """
        super().__init__()
        d = 9 if isMatrix else 6
        n_imu = num_joint*3 + num_joint * d   # acceleration (vector3) and rotation matrix (matrix3x3) of 6 IMUs
        if onlyori:
            n_imu -= num_joint *3
        self.pose_net = RNN_SRU(n_imu,                         joint_set.n_reduced * 6,       256, n_rnn_layer=5)
        # lower body joint
        self.m = art.ParametricModel(paths.male_smpl_file, device=device)
        j, _ = self.m.get_zero_pose_joint_and_vertex()
        b = art.math.joint_position_to_bone_vector(j[joint_set.lower_body].unsqueeze(0),
                                                   joint_set.lower_body_parent).squeeze(0)
        bone_orientation, bone_length = art.math.normalize_tensor(b, return_norm=True)
        if hip_length is not None:
            bone_length[1:3] = torch.tensor(hip_length)
        if upper_leg_length is not None:
            bone_length[3:5] = torch.tensor(upper_leg_length)
        if lower_leg_length is not None:
            bone_length[5:7] = torch.tensor(lower_leg_length)
        b = bone_orientation * bone_length
        b[:3] = 0

        # constant
        self.global_to_local_pose = self.m.inverse_kinematics_R
        self.lower_body_bone = b
        self.num_past_frame = num_past_frame
        self.num_future_frame = num_future_frame
        self.num_total_frame = num_past_frame + num_future_frame + 1
        self.prob_threshold = prob_threshold
        self.gravity_velocity = torch.tensor([0, gravity_velocity, 0]).to(device)
        self.feet_pos = j[10:12].clone()
        self.floor_y = j[10:12, 1].min().item()

        # variable
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_lfoot_pos, self.last_rfoot_pos = self.feet_pos
        self.last_root_pos = torch.zeros(3)
        
        self.poseFilter_online = PoseFilter(smooth_alpha) if smooth_alpha!=-1 else None
        self.poseFilter_offline = PoseFilter(smooth_alpha) if smooth_alpha!=-1 else None
        self.reset()

    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose, filter=None):
        if filter:glb_reduced_pose = torch.stack([filter.update(i) for i in glb_reduced_pose.view(-1, 15, 6)])
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = root_rotation.view(-1, 3, 3)
        return pose
        
    def reset(self):
        r"""
        Reset online forward states.
        """
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_lfoot_pos, self.last_rfoot_pos = self.feet_pos
        self.last_root_pos = torch.zeros(3)
        if self.poseFilter_offline:
            self.poseFilter_offline.reset()
            self.poseFilter_online.reset()
        

    def forward(self, imu, rnn_state=None):
        global_reduced_pose, _ = self.pose_net.forward(imu)
        contact_probability = None
        return global_reduced_pose, contact_probability, None, rnn_state
    
    def forward_my(self, input, rnn_state=None, refine=False):
        imu, leaf_jtr, full_jtr =  input
        if not refine: 
            imu += torch.normal(mean=imu, std=0.04).to(imu.device)
        global_reduced_pose = self.pose_net.forward(imu)[0]

        contact_prob = None
        return global_reduced_pose, contact_prob, None, rnn_state

    @torch.no_grad()
    def forward_offline(self, x):
        r"""
        Offline forward.

        :param imu: Tensor in shape [num_frame, input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [num_frame, 24, 3, 3] and velocity tensor in shape [num_frame, 3].
        """
        imu = x
        root_rotation = x[:, 0, -6:]
        root_rotation = art.math.r6d_to_rotation_matrix(root_rotation)
        
        global_reduced_pose, contact_probability, velocity, _ = self.forward(imu) 

        pose = self._reduced_glb_6d_to_full_local_mat(root_rotation, global_reduced_pose, self.poseFilter_offline)
    
        return pose, None

    @torch.no_grad()
    def forward_online(self, x):
        r"""
        Online forward.

        :param x: A tensor in shape [input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [24, 3, 3] and velocity tensor in shape [3].
        """
        imu = x.repeat(self.num_total_frame, 1, 1) if self.imu is None else torch.cat((self.imu[1:], x.view(1, 1, -1)))
        global_reduced_pose, _, _, rnn_state = self.forward(imu, self.rnn_state)

        # calculate pose (local joint rotation matrices)
        root_rotation = imu[self.num_past_frame, 0, -6:]
        root_rotation = art.math.r6d_to_rotation_matrix(root_rotation)
        
        global_reduced_pose = global_reduced_pose[self.num_past_frame]
        pose = self._reduced_glb_6d_to_full_local_mat(root_rotation, global_reduced_pose, self.poseFilter_online).squeeze(0)

        self.rnn_state = rnn_state
        self.imu = imu
        return pose, torch.zeros((len(pose), 3))
