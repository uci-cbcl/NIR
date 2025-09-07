import enum
import matplotlib.pyplot as plt
import numpy as np
import nrrd
import torch
import dataio
import os
import diff_operators
from torchvision.utils import make_grid, save_image
from skimage import measure
from skimage.metrics import structural_similarity as ssim
import cv2
from scipy.spatial.distance import cdist
import scipy.io.wavfile as wavfile
from scipy.ndimage.interpolation import map_coordinates

def min_max_summary(name, tensor, writer, total_steps):
    writer.add_scalar(name + '_min', tensor.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(name + '_max', tensor.max().detach().cpu().numpy(), total_steps)

def write_3d_img_summary(img_dataset, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    resolution = np.asarray(img_dataset.shape)
    frames = [20, 60, 100, 140]
    Nslice = 10
    with torch.no_grad():
        coords = []
        for f in frames:
            coord, maxlen = dataio.get_mgrid((1, resolution[1], resolution[2]), img_dataset.maxlen, dim=3)
            coord = dataio.normalize_coords(coord.reshape(-1, 3), resolution, maxlen)
            coord[..., 0] = ((f - (resolution[0]-1)/2) / (maxlen[0]-1)) * 2
            coords += [coord[None,...].cuda()]
        coords = torch.cat(coords, dim=0)

        output = torch.zeros(coords.shape[:-1] + (3, ))
        weight_masks = torch.zeros(coords.shape[:-1] + (3, ))
        split = int(coords.shape[1] / Nslice)
        for i in range(Nslice):
            model_output = model({'coords':coords[:, i*split:(i+1)*split, :]})
            pred = model_output['model_out']
            if 'residual_weights' in model_output:
                residual_weights = model_output['residual_weights']
                weight_masks[:, i*split:(i+1)*split, :] = residual_weights.cpu()
            else:
                residual_weights = None
            output[:, i*split:(i+1)*split, :] = pred.cpu()

    pred_img = output.view(len(frames), resolution[1], resolution[2], -1) / 2 + 0.5
    pred_img = torch.clamp(pred_img, 0, 1)
    gt_img = torch.from_numpy(img_dataset[0][frames, :, :])
    psnr = 10*torch.log10(1 / torch.mean((gt_img - pred_img)**2))

    pred_img = pred_img.permute(0, 3, 1, 2)
    gt_img = gt_img.permute(0, 3, 1, 2).expand_as(pred_img)

    output_vs_gt = torch.cat((gt_img, pred_img), dim=-2)

    if residual_weights is not None:
        weight_masks = weight_masks.view(len(frames), resolution[1], resolution[2], -1)
        weight_masks = weight_masks.permute(0, 3, 1, 2)
        output_vs_gt = torch.cat((output_vs_gt, weight_masks), dim=-2)

    writer.add_image(prefix + 'output_vs_gt', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)
    min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
    min_max_summary(prefix + 'pred_img', pred_img, writer, total_steps)
    writer.add_scalar(prefix + "psnr", psnr, total_steps)


def write_3d_img_pairwise_reg_summary(img_dataset, deform_model, moving_image_model, writer, total_steps, prefix='train_'):
    resolution = np.asarray(img_dataset.shape)
    frames = [20, 60, 100, 140]
    Nslice = 10
    with torch.no_grad():
        coords = []
        for f in frames:
            coord, maxlen = dataio.get_mgrid((1, resolution[1], resolution[2]), img_dataset.maxlen, dim=3)
            coord = dataio.normalize_coords(coord, resolution, maxlen)
            coord[..., 0] = ((f - (resolution[0]-1)/2) / (maxlen[0]-1)) * 2
            coords += [coord.cuda()]
        coords = torch.cat(coords, dim=1)

        output = torch.zeros(coords.shape[:-1] + (img_dataset.channels, ))
        moving = torch.zeros(coords.shape[:-1] + (img_dataset.channels, ))
        for i in range(len(frames)):
            moving[:, i:i+1, ...] = moving_image_model({'coords':coords[:, i:i+1, ...]})['model_out'].cpu()
            output[:, i:i+1, ...] = deform_model({'coords':coords[:, i:i+1, ...]}, moving_image_model)['model_out'].cpu()

            # deform_output = deform_model({'coords':coords[:, i*split:(i+1)*split, :]})
            # deformation = deform_output['model_out'][:,:,:3]
            # new_coords = deform_output['model_in'] + deformation
            # output[:, i*split:(i+1)*split, :] = moving_image_model({'coords':new_coords})['model_out'].cpu()

    pred_img = output.view(len(frames), resolution[1], resolution[2], -1) / 2 + 0.5
    pred_img = torch.clamp(pred_img, 0, 1)
    gt_img = torch.from_numpy(img_dataset[0][frames, :, :])
    moving_img = moving.view(len(frames), resolution[1], resolution[2], -1) / 2 + 0.5
    moving_img = torch.clamp(moving_img, 0, 1)
    psnr = 10*torch.log10(1 / torch.mean((gt_img - pred_img)**2))

    pred_img = pred_img.permute(0, 3, 1, 2)
    gt_img = gt_img.permute(0, 3, 1, 2)
    moving_img = moving_img.permute(0, 3, 1, 2)

    output_vs_gt = torch.cat((gt_img, pred_img, moving_img), dim=-2)
    writer.add_image(prefix + 'output_vs_gt', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)
    writer.add_scalar(prefix + "psnr", psnr, total_steps)


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_mask(mask_dir, pid, roi_names):
    mask = []
    for roi_name in roi_names:
        mask += [nrrd.read(os.path.join(mask_dir, pid, roi_name + '.nrrd'))[0]]
    return mask


def warp_3d_volume(volume, deformed_coords, normalization_ratio, order=3, cval=0):
    raw_deformed_coords = deformed_coords / 2 * normalization_ratio + (np.asarray(volume.shape)[None, None, None,:] - 1) / 2
    warped_volume = map_coordinates(volume, raw_deformed_coords.transpose((3,0,1,2)), order=order, mode='constant', cval=cval)
    return warped_volume


def dice_score(m, p):
    m = m.astype(bool).astype(np.uint8)
    p = p.astype(bool).astype(np.uint8)
    intersect = float(2 * (m * p).sum())
    denominator = float(m.sum() + p.sum())

    score = intersect / denominator
    
    return score


def negative_jacobian_det(new_coords, maxlen, mask=None):
    grad_dxyzdx = (new_coords[1:, :-1, :-1] - new_coords[:-1, :-1, :-1]) * (maxlen[0] / 2)
    grad_dxyzdy = (new_coords[:-1, 1:, :-1] - new_coords[:-1, :-1, :-1]) * (maxlen[1] / 2)
    grad_dxyzdz = (new_coords[:-1, :-1, 1:] - new_coords[:-1, :-1, :-1]) * (maxlen[2] / 2)
    jacobian = np.stack([grad_dxyzdx, grad_dxyzdy, grad_dxyzdz], axis=-1)  # [3, B, N, 3]
    jacobiran_det = np.linalg.det(jacobian)
    if mask is None:
        num_neg_jdet = np.sum(jacobiran_det < 0)
    else:
        num_neg_jdet = np.sum((jacobiran_det * mask[1:,1:,1:]) < 0)
    return float(num_neg_jdet)

def ssim_score(warped_img, img, data_range):
    return float(ssim(img, warped_img, data_range=data_range, multichannel=False))

# def get_contours_from_mask(mask):
#     """
#     Generate contours from masks by going through each organ slice by slice
#     masks: [D, H, W]
#     return: contours of shape [D, H, W]
#     """
#     contours = np.zeros(mask.shape, dtype=np.uint8)

#     point_z = {}
#     num_points = 0
#     # For each organ, Iterate all slices
#     for j, s in enumerate(mask):
#         pts = measure.find_contours(s, 0)

#         if pts:
#             # There is contour in the image
#             pts = np.concatenate(pts).astype(np.int32)
#             num_points += len(pts)
#             point_z[j] = pts

#     if num_points > 100000:
#         for j in point_z:
#             pts = point_z[j]
#             for k, point in enumerate(pts):
#                 if k % 4 == 0:
#                     contours[j, point[0], point[1]] = 1
#     elif num_points > 50000:
#         for j in point_z:
#             pts = point_z[j]
#             for k, point in enumerate(pts):
#                 if k % 2 == 0:
#                     contours[j, point[0], point[1]] = 1
#     else:
#         for j in point_z:
#             pts = point_z[j]
#             for k, point in enumerate(pts):
#                 contours[j, point[0], point[1]] = 1

#     return contours

# def hd95_and_surface_dice(mask, pred, spacing):
#     percent = 0.95
#     mask = get_contours_from_mask(mask)
#     pred = get_contours_from_mask(pred)

#     a_pts = np.where(mask)
#     b_pts = np.where(pred)
#     a_pts = np.array(a_pts).T * np.array(spacing).astype(np.float32)
#     b_pts = np.array(b_pts).T * np.array(spacing).astype(np.float32)

#     dists = cdist(a_pts, b_pts)

#     del a_pts, b_pts

#     a = np.min(dists, 1).astype(np.float32)
#     b = np.min(dists, 0).astype(np.float32)
#     a.sort()
#     b.sort()

#     a_max = a[int(percent * len(a)) - 1]
#     b_max = b[int(percent * len(b)) - 1]

#     score = max(a_max, b_max)
#     return float(score)

import surface_distance

def hd95(mask, pred, spacing_mm):
    surface_distances = surface_distance.compute_surface_distances(
            mask, pred, spacing_mm=spacing_mm)
    hd95_score = surface_distance.compute_robust_hausdorff(surface_distances, 95)
    surface_dice_score = surface_distance.compute_surface_dice_at_tolerance(
            surface_distances, tolerance_mm=1)
    
    return float(hd95_score), float(surface_dice_score)
