import torch
import torch.nn.functional as F

from misc.chamfer_distance import ChamferDistance

def chamfer_mask(x, y_hat0, y_hat1, thres_static=0.2, thres_dist=0.1):
    # x: B T N C
    # y_hat0: B 1 J 3
    # y_hat1: B 1 J 3

    x_t01 = x[:, -2:, ...]
    x_t0 = x_t01[:, 0:1, :, :3]  # B 1 N 3
    x_t1 = x_t01[:, 1:2, :, :3]  # B 1 N 3

    chamfer_dist = ChamferDistance()
    dist1, dist2 = chamfer_dist(x_t0[:, 0].to(torch.float), y_hat0[:, 0].to(torch.float))

    mask_dist_pos = (dist2 < thres_dist).unsqueeze(1).unsqueeze(-1).detach()  # B 1 J 1
    mask_dist_neg = (dist2 > thres_static).unsqueeze(1).unsqueeze(-1).detach()  # B 1 J 1
    return mask_dist_pos, mask_dist_neg

class UnsupLoss(torch.nn.Module):
    def __init__(self, thres_static=0.2, thres_dist=0.1):
        super().__init__()
        self.thres_static = thres_static
        self.thres_dist = thres_dist
        self.chamfer_dist = ChamferDistance()

    def forward(self, x, y_hat0, y_hat1):
        # x: B T N C
        # y_hat0: B 1 J 3
        # y_hat1: B 1 J 3

        mask_dist_pos, mask_dist_neg = chamfer_mask(x, y_hat0, y_hat1, self.thres_static, self.thres_dist)
        
        my_hat_dynamic = (y_hat1 - y_hat0) * mask_dist_pos
        my_norm_hat_dynamic = torch.norm(my_hat_dynamic, p=2, dim=-1)
        my_norm_hat_dynamic = my_norm_hat_dynamic[my_norm_hat_dynamic > 0]
        loss_dynamic = torch.nn.functional.softplus(0.1 - my_norm_hat_dynamic).mean()
        
        my_hat_static = (y_hat1 - y_hat0) * mask_dist_neg
        my_hat_static = my_hat_static[my_hat_static > 0]
        loss_static = F.mse_loss(my_hat_static, torch.zeros_like(my_hat_static))

        return loss_dynamic, loss_static