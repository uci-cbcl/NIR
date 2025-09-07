# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from src import dataio, util, modules, deform_net

from torch.utils.data import DataLoader
import numpy as np
import nrrd
import yaml
import configargparse
import torch

p = configargparse.ArgumentParser()
p.add_argument('--logging_root', type=str, default='./logs_HLN-12-1', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--image_dir', type=str, help='3D Medical moving Image dir')
p.add_argument('--mask_dir', type=str, help='3D Medical moving Mask dir')
p.add_argument('--fixed_pid', type=str, help='3D Medical fixed Image patient id')
p.add_argument('--moving_pid', type=str, help='3D Medical moving Image patient id')
p.add_argument('--epoch_name', type=str, required=True, help='in the format of epoch_*')
opt = p.parse_args()

roi_names = ['Left-Cerebral-White-Matter', 'Left-Cerebral-Cortex', 'Left-Lateral-Ventricle', 'Left-Cerebellum-White-Matter', 'Left-Cerebellum-Cortex', 'Left-Thalamus', 'Left-Caudate',
            'Left-Putamen', 'Left-Pallidum', '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'Left-Ventral-DC',
            'Right-Cerebral-White-Matter', 'Right-Cerebral-Cortex', 'Right-Lateral-Ventricle', 'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex', 'Right-Thalamus',
            'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala', 'Right-Ventral-DC']

eval_save_dir = os.path.join(opt.logging_root, 'reg', 'eval_yaml_results')
os.makedirs(eval_save_dir, exist_ok=True)

results_dir = os.path.join(opt.logging_root, 'reg', 
                           f'from_{opt.moving_pid}_to_{opt.fixed_pid}', 
                           opt.experiment_name, f'results_{opt.epoch_name}')

if opt.mask_dir != 'empty':
    if not os.path.exists(os.path.join(eval_save_dir, f"{opt.fixed_pid}_dice_eval_results.yaml")):
        with open(os.path.join(eval_save_dir, f"{opt.fixed_pid}_dice_eval_results.yaml"), 'w') as file:
            documents = yaml.dump(dict(), file)

    if not os.path.exists(os.path.join(eval_save_dir, f"{opt.fixed_pid}_surface_dice_eval_results.yaml")):
        with open(os.path.join(eval_save_dir, f"{opt.fixed_pid}_surface_dice_eval_results.yaml"), 'w') as file:
            documents = yaml.dump(dict(), file)

    if not os.path.exists(os.path.join(eval_save_dir, f"{opt.fixed_pid}_hd95_eval_results.yaml")):
        with open(os.path.join(eval_save_dir, f"{opt.fixed_pid}_hd95_eval_results.yaml"), 'w') as file:
            documents = yaml.dump(dict(), file)

    with open(os.path.join(eval_save_dir, f"{opt.fixed_pid}_dice_eval_results.yaml")) as file:
        dice_eval_results_dict = yaml.load(file, Loader=yaml.FullLoader) or {}

    with open(os.path.join(eval_save_dir, f"{opt.fixed_pid}_surface_dice_eval_results.yaml")) as file:
        surface_dice_eval_results_dict = yaml.load(file, Loader=yaml.FullLoader) or {}

    with open(os.path.join(eval_save_dir, f"{opt.fixed_pid}_hd95_eval_results.yaml")) as file:
        hd95_eval_results_dict = yaml.load(file, Loader=yaml.FullLoader) or {}

    if opt.epoch_name not in dice_eval_results_dict:
        dice_eval_results_dict[opt.epoch_name] = dict()

    if opt.epoch_name not in surface_dice_eval_results_dict:
        surface_dice_eval_results_dict[opt.epoch_name] = dict()

    if opt.epoch_name not in hd95_eval_results_dict:
        hd95_eval_results_dict[opt.epoch_name] = dict()

    if opt.moving_pid not in dice_eval_results_dict[opt.epoch_name]:
        dice_eval_results_dict[opt.epoch_name][opt.moving_pid] = dict()

    if opt.moving_pid not in surface_dice_eval_results_dict[opt.epoch_name]:
        surface_dice_eval_results_dict[opt.epoch_name][opt.moving_pid] = dict()

    if opt.moving_pid not in hd95_eval_results_dict[opt.epoch_name]:
        hd95_eval_results_dict[opt.epoch_name][opt.moving_pid] = dict()

    for i, roi_name in enumerate(roi_names):

        if roi_name not in dice_eval_results_dict[opt.epoch_name][opt.moving_pid]:
            dice_eval_results_dict[opt.epoch_name][opt.moving_pid][roi_name] = dict()

        if roi_name not in surface_dice_eval_results_dict[opt.epoch_name][opt.moving_pid]:
            surface_dice_eval_results_dict[opt.epoch_name][opt.moving_pid][roi_name] = dict()

        if roi_name not in hd95_eval_results_dict[opt.epoch_name][opt.moving_pid]:
            hd95_eval_results_dict[opt.epoch_name][opt.moving_pid][roi_name] = dict()

if not os.path.exists(os.path.join(eval_save_dir, f"{opt.fixed_pid}_njdet_eval_results.yaml")):
    with open(os.path.join(eval_save_dir, f"{opt.fixed_pid}_njdet_eval_results.yaml"), 'w') as file:
        documents = yaml.dump(dict(), file)
if not os.path.exists(os.path.join(eval_save_dir, f"{opt.fixed_pid}_ssim_eval_results.yaml")):
    with open(os.path.join(eval_save_dir, f"{opt.fixed_pid}_ssim_eval_results.yaml"), 'w') as file:
        documents = yaml.dump(dict(), file)

with open(os.path.join(eval_save_dir, f"{opt.fixed_pid}_njdet_eval_results.yaml")) as file:
    njdet_eval_results_dict = yaml.load(file, Loader=yaml.FullLoader) or {}
with open(os.path.join(eval_save_dir, f"{opt.fixed_pid}_ssim_eval_results.yaml")) as file:
    ssim_eval_results_dict = yaml.load(file, Loader=yaml.FullLoader) or {}

# eval_results_dict = dict()
if opt.epoch_name not in njdet_eval_results_dict:
    njdet_eval_results_dict[opt.epoch_name] = dict()
if opt.epoch_name not in ssim_eval_results_dict:
    ssim_eval_results_dict[opt.epoch_name] = dict()

if opt.moving_pid not in njdet_eval_results_dict[opt.epoch_name]:
    njdet_eval_results_dict[opt.epoch_name][opt.moving_pid] = dict()
if opt.moving_pid not in ssim_eval_results_dict[opt.epoch_name]:
    ssim_eval_results_dict[opt.epoch_name][opt.moving_pid] = dict()

gt_img_dir = os.path.join(opt.image_dir, opt.fixed_pid)
gt_img = nrrd.read(os.path.join(gt_img_dir, 'img.nrrd'))[0]
warped_moving_img = nrrd.read(os.path.join(results_dir, 'warped_img.nrrd'))[0]
ssim = util.ssim_score(warped_moving_img, gt_img, gt_img.max() - gt_img.min())
ssim_eval_results_dict[opt.epoch_name][opt.moving_pid][opt.experiment_name] = ssim
with open(os.path.join(eval_save_dir, f"{opt.fixed_pid}_ssim_eval_results.yaml"), 'w') as file:
    documents = yaml.dump(ssim_eval_results_dict, file)

new_coords = np.load(os.path.join(results_dir, 'new_coords.npy'))
neg_jdet = util.negative_jacobian_det(new_coords, new_coords.shape, gt_img > 0)
njdet_eval_results_dict[opt.epoch_name][opt.moving_pid][opt.experiment_name] = neg_jdet
with open(os.path.join(eval_save_dir, f"{opt.fixed_pid}_njdet_eval_results.yaml"), 'w') as file:
    documents = yaml.dump(njdet_eval_results_dict, file)

if opt.mask_dir != 'empty':
    dice_scores = []
    hd95_scores = []
    surface_dice_scores = []
    gt_mask_dir = os.path.join(opt.mask_dir, opt.fixed_pid)
    for i, roi_name in enumerate(roi_names):
        gt_mask = nrrd.read(os.path.join(gt_mask_dir, f'{roi_name}.nrrd'))[0]
        warped_moving_mask = nrrd.read(os.path.join(results_dir, f'warped_{roi_name}.nrrd'))[0]
        dice_score = util.dice_score(gt_mask, warped_moving_mask)

        dice_eval_results_dict[opt.epoch_name][opt.moving_pid][roi_name][opt.experiment_name] = dice_score
        dice_scores += [dice_score]

        hd95_score, surface_dice_score = util.hd95(gt_mask.astype(bool), warped_moving_mask.astype(bool), np.asarray([1.,1.,1.]))
        hd95_eval_results_dict[opt.epoch_name][opt.moving_pid][roi_name][opt.experiment_name] = hd95_score
        surface_dice_eval_results_dict[opt.epoch_name][opt.moving_pid][roi_name][opt.experiment_name] = surface_dice_score
        hd95_scores += [hd95_score]
        surface_dice_scores += [surface_dice_score]

    with open(os.path.join(eval_save_dir, f"{opt.fixed_pid}_dice_eval_results.yaml"), 'w') as file:
        documents = yaml.dump(dice_eval_results_dict, file)

    with open(os.path.join(eval_save_dir, f"{opt.fixed_pid}_surface_dice_eval_results.yaml"), 'w') as file:
        documents = yaml.dump(surface_dice_eval_results_dict, file)

    with open(os.path.join(eval_save_dir, f"{opt.fixed_pid}_hd95_eval_results.yaml"), 'w') as file:
        documents = yaml.dump(hd95_eval_results_dict, file)


if opt.mask_dir != 'empty':
    print(f"Finish {opt.epoch_name} {opt.fixed_pid}: {opt.moving_pid}, dice score: {np.mean(dice_scores)}, hd95 score: {np.mean(hd95_scores)}, surface_dice: {np.mean(surface_dice_scores)}, neg_jdet: {neg_jdet}, ssim: {ssim}")
else:
    print(f"Finish {opt.epoch_name} {opt.fixed_pid}: {opt.moving_pid}, neg_jdet: {neg_jdet}, ssim: {ssim}")