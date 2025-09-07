from os import pread
from tokenize import group
from xml.parsers.expat import model
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
import torch.fft as fft
import torch.nn.functional as F
from math import exp
import diff_operators


def image_mse(mask, model_output, gt):
    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean()}


def image_l1(mask, model_output, gt):
    if mask is None:
        return {'img_loss': torch.abs(model_output['model_out'] - gt['img']).mean()}
    else:
        return {'img_loss': (mask * torch.abs(model_output['model_out'] - gt['img'])).mean()}


def ncc(model_output, gt): 
    if 'win_scale' in gt:
        win_size = int(18 / gt['win_scale'][0])
    else:
        win_size = 9
    y_pred = model_output['model_out'] / 2 + 0.5
    y_true = gt['img'] / 2 + 0.5

    B, channel = y_pred.shape[0], y_pred.shape[-1]
    if 'patch_len' in gt:
        patch_len = gt['patch_len']

        y_pred = y_pred.reshape(B, patch_len, patch_len, patch_len, channel)
        y_true = y_true.reshape(B, patch_len, patch_len, patch_len, channel)
        
    y_pred = y_pred.permute((0,4,1,2,3))
    y_true = y_true.permute((0,4,1,2,3))

    sum_filt = torch.ones([channel, 1, win_size, win_size, win_size]).to(y_true)
    y_pred_sum = F.conv3d(y_pred, sum_filt, stride=1, padding=0, groups=channel)
    y_true_sum = F.conv3d(y_true, sum_filt, stride=1, padding=0, groups=channel)

    ############################################  Key Design, otherwise, gradient will explode.  #########################################################################################################################################################
    # for the zero part, use mse loss as the auxiliary loss, otherwise, the gradient will explode.
    y_zero_flag = y_pred_sum.le(0) + y_true_sum.le(0)
    zero_weight = torch.ones([channel, 1, win_size, win_size, win_size]).to(y_true)
    y_zero_flag = F.conv_transpose3d(y_zero_flag.float(), zero_weight, stride=1, padding=0).gt(0)
    if y_zero_flag.sum() == 0:
        auxiliary_mse_loss = 0
    else:
        auxiliary_mse_loss = torch.mean(torch.dropout(y_pred[y_zero_flag.expand_as(y_pred)] - y_true[y_zero_flag.expand_as(y_true)], p=0.2, train=True) ** 2)
    ############################################################################################################################################################################################################################

    # for the non zero part, calculate the NCC loss
    y_non_zero_flag = y_pred_sum.gt(0) * y_true_sum.gt(0)
    y_pred_2 = y_pred * y_pred
    y_true_2 = y_true * y_true
    y_pred_true = y_pred * y_true
    y_pred_2_sum = F.conv3d(y_pred_2, sum_filt, stride=1, padding=0, groups=channel)
    y_true_2_sum = F.conv3d(y_true_2, sum_filt, stride=1, padding=0, groups=channel)
    y_pred_true_sum = F.conv3d(y_pred_true, sum_filt, stride=1, padding=0, groups=channel)

    win_size = win_size**3
    u_y_pred = y_pred_sum / win_size
    u_y_true = y_true_sum / win_size

    cross = y_pred_true_sum - u_y_true * y_pred_sum - u_y_pred * y_true_sum + u_y_pred * u_y_true * win_size
    y_pred_var = y_pred_2_sum - 2 * u_y_pred * y_pred_sum + u_y_pred * u_y_pred * win_size
    y_true_var = y_true_2_sum - 2 * u_y_true * y_true_sum + u_y_true * u_y_true * win_size

    cc = cross * cross / (y_pred_var * y_true_var + 1e-5) * y_non_zero_flag.float()
    cc = torch.dropout(cc, p=0.2, train=True)

    cc_loss = -torch.sum(cc) / (torch.sum(y_non_zero_flag)+1e-6)

    return {'ncc': auxiliary_mse_loss + cc_loss}
    

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window =_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window

    
def ssim_compute(img1, img2, window, window_size, channel):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    return ssim_map


def ssim(model_output, gt):
    win_size = 9
    y_pred = model_output['model_out'] / 2 + 0.5
    y_true = gt['img'] / 2 + 0.5

    B, channel = y_pred.shape[0], y_pred.shape[-1]

    if 'patch_len' in gt:
        patch_len = gt['patch_len']

        y_pred = y_pred.reshape(B, patch_len, patch_len, patch_len, channel)
        y_true = y_true.reshape(B, patch_len, patch_len, patch_len, channel)
        
    y_pred = y_pred.permute((0,4,1,2,3))
    y_true = y_true.permute((0,4,1,2,3))

    sum_filt = torch.ones([1, 1, win_size, win_size, win_size]).to(y_true)
    y_pred_sum = F.conv3d(y_pred, sum_filt, stride=1, padding=0)
    y_true_sum = F.conv3d(y_true, sum_filt, stride=1, padding=0)

    ############################################  Key Design, otherwise, gradient will explode.  #########################################################################################################################################################
    # for the zero part, use mse loss as the auxiliary loss, otherwise, the gradient will explode.
    y_zero_flag = y_pred_sum.le(0.1) + y_true_sum.le(0.1)
    zero_weight = torch.ones([1, 1, win_size, win_size, win_size]).to(y_true)
    y_zero_flag = F.conv_transpose3d(y_zero_flag.float(), zero_weight, stride=1, padding=0).gt(0)
    if y_zero_flag.sum() == 0:
        auxiliary_mse_loss = 0
    else:
        auxiliary_mse_loss = torch.mean(torch.dropout(y_pred[y_zero_flag] - y_true[y_zero_flag], p=0.2, train=True) ** 2)
    ############################################################################################################################################################################################################################
    # for the non zero part, calculate the NCC loss
    y_non_zero_flag = y_pred_sum.gt(-0.1) * y_true_sum.gt(-0.1)
    y_non_zero_flag = F.conv_transpose3d(y_non_zero_flag.float(), zero_weight, stride=1, padding=0).gt(0)

    window = create_window(win_size, channel).to(y_true)
    ssim_loss = ssim_compute(y_true, y_pred, window, win_size, channel) * y_non_zero_flag
    ssim_loss = -torch.sum(ssim_loss) / (torch.sum(y_non_zero_flag)+1e-6)

    return {'ssim': auxiliary_mse_loss + ssim_loss}


def discrete_jacobian_det(model_output, gt, maxlen, weights=10):
    if 'win_scale' in gt:
        win_scale = gt['win_scale'][0]
    else:
        win_scale = 3
    new_coords = model_output['new_coords']
    grad_dxyzdx = (new_coords[:, 1:, :-1, :-1] - new_coords[:, :-1, :-1, :-1]) * (maxlen[0] / (2*win_scale))
    grad_dxyzdy = (new_coords[:, :-1, 1:, :-1] - new_coords[:, :-1, :-1, :-1]) * (maxlen[1] / (2*win_scale))
    grad_dxyzdz = (new_coords[:, :-1, :-1, 1:] - new_coords[:, :-1, :-1, :-1]) * (maxlen[2] / (2*win_scale))
    jacobian = torch.stack([grad_dxyzdx, grad_dxyzdy, grad_dxyzdz], dim=-1)  # [3, B, N, 3]
    jacobian_reg = F.relu(-torch.linalg.det(jacobian))
    # jacobian_reg = torch.abs(1-torch.linalg.det(jacobian))
    return {'discrete_jacobian_reg': weights * jacobian_reg.mean()}
