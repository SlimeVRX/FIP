import torch

class MyLoss1Stage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l2lLoss = torch.nn.SmoothL1Loss()

    def forward(self, x, y, refine):
        global_reduced_pose, contact_probability, velocity, _ = x
        global_reduced_pose_gt, contact_probability_gt, velocity_gt = y

        poseLoss = self.l2lLoss(global_reduced_pose, global_reduced_pose_gt)

        loss_dict =  {"pose":poseLoss}

        
        return loss_dict
       