# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from src import dataio, util, modules, efficient_modules, deform_net

from torch.utils.data import DataLoader
import numpy as np
import nrrd
import configargparse
import torch

p = configargparse.ArgumentParser()
p.add_argument('--logging_root', type=str, default='./logs_HLN-12-1', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=25,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=1000,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options currently are "sine" (all sine activations), "relu" (all relu activations,'
                    '"nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu),'
                    'and in the future: "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--gaussian_scale', type=float, default=5, help='Gaussian sclae for position encoding')
p.add_argument('--num_hidden_layers', type=int, default=3, help='Number of hidden layers in the mlps')
p.add_argument('--image_dir', type=str, help='3D Medical moving Image dir')
p.add_argument('--mask_dir', type=str, help='3D Medical moving Mask dir')
p.add_argument('--fixed_pid', type=str, help='3D Medical fixed Image patient id')
p.add_argument('--moving_pid', type=str, help='3D Medical moving Image patient id')
p.add_argument('--epoch_name', type=str, required=True, help='in the format of epoch_*')
opt = p.parse_args()

roi_names = ['Left-Cerebral-White-Matter', 'Left-Lateral-Ventricle', 'Right-Cerebellum-Cortex', 'Right-Accumbens', '4th-Ventricle', 
             'Right-Lateral-Ventricle', 'Right-Cerebral-Cortex', 'Right-Cerebral-White-Matter', 'Left-Cerebral-Cortex', 'Right-Thalamus', 
             'Right-Inf-Lat-Ventricle', 'Left-Choroid-Plexus', 'Left-Hippocampus', 'Left-Putamen', 'Right-Caudate', 'Left-Pallidum', 'Right-Putamen', 
             'Right-Cerebellum-White-Matter', 'Left-Accumbens', 'foreground', 'Left-Cerebellum-Cortex', 'Right-Ventral-DC', 'Right-Amygdala', 
             '3rd-Ventricle', 'Right-Vessel', 'Left-Cerebellum-White-Matter', 'Right-Hippocampus', 'Brain-Stem', 'Left-Thalamus', 'Left-Vessel', 
             'Right-Pallidum', 'Right-Choroid-Plexus', 'Left-Amygdala', 'Left-Caudate', 'Left-Inf-Lat-Ventricle', 'Left-Ventral-DC']

ct_dataset = dataio.Single_CT_Brain(os.path.join(opt.image_dir, opt.moving_pid), maxlen=None)
coord_dataset = dataio.Implicit3DWrapper(ct_dataset, sample_method='whole')
dataloader = DataLoader(coord_dataset, shuffle=False, batch_size=opt.batch_size, pin_memory=True, num_workers=0)
ct_dataset.maxlen = coord_dataset.maxlen

if opt.checkpoint_path is None:
    opt.checkpoint_path = os.path.join(opt.logging_root, 'reg', f'from_{opt.moving_pid}_to_{opt.fixed_pid}', 
                                       opt.experiment_name, 'checkpoints', f'model_{opt.epoch_name}.pth')
                                       
# Define the model.
if 'ode' in opt.model_type:
    # fourier_positional_encoding = modules.PosEncodingFourier(in_features=3, method='gaussian', 
    #                                                         gaussian_scale=opt.gaussian_scale, embedding_size=128)
    if "gaussian" in opt.model_type:
        positional_encoding = efficient_modules.PosEncodingFourier(in_features=3, method='gaussian', 
                                                            gaussian_scale=opt.gaussian_scale, embedding_size=128)
        in_features = 256
    elif "hash" in opt.model_type:
        positional_encoding = efficient_modules.HashEncoding(3, 128, 16, 8, n_features_per_level=8, log2_hashmap_size=12, dtype=torch.float)
        in_features = 64
    dynamic_net = modules.FCBlock(in_features=in_features, out_features=3, num_hidden_layers=opt.num_hidden_layers, 
                                  hidden_features=256, outermost_linear=True, nonlinearity=opt.model_type.split('_')[-1])
    deform_field_model = deform_net.ODEFunc(positional_encoding, dynamic_net)
    reg_model = deform_net.DiffeoDeformedField(deform_field_model)
elif opt.model_type == 'grid':
    whole_coords = torch.zeros((1,) + ct_dataset.shape + (3,))
    reg_model = deform_net.DeformedGridField(whole_coords, 'residual')
else:
    deform_field_model = efficient_modules.EfficientMLP(in_features=3, out_features=len(ct_dataset.shape), num_hidden_layers=opt.num_hidden_layers,
                                                        mode=opt.model_type.split('_')[0], gaussian_scale=opt.gaussian_scale, nonlinearity=opt.model_type.split('_')[-1])
    
    reg_model = deform_net.DeformedField(deform_field_model, 'residual')

reg_model.load_state_dict(torch.load(opt.checkpoint_path))
reg_model.cuda()

# Get ground truth and input data
model_input, _ = next(iter(dataloader))
coords_all = model_input['coords']

if opt.model_type == 'grid':
    new_coords = reg_model.deform(coords_all.cuda()).cpu()
else:
    coords_chunk = torch.chunk(coords_all, 6, 1)

    # Evaluate the trained model
    new_coords = []
    # model_outputs = []
    with torch.no_grad():
        for coords in coords_chunk:
            coords_shape = coords.shape
            with torch.cuda.amp.autocast(enabled=False):
                model_output = reg_model.deform(coords.reshape(1, -1, 3).cuda())
            new_coords += [model_output.reshape(coords_shape).cpu()]
            # model_outputs += [reg_model({'coords':coords.cuda()}, None)['model_out'].cpu()]

    new_coords = torch.cat(new_coords, dim=1)

results_dir = os.path.join(opt.logging_root, 'reg', 
                           f'from_{opt.moving_pid}_to_{opt.fixed_pid}', 
                           opt.experiment_name, f'results_{opt.epoch_name}')
raw_coords = coords_all[0].numpy()
new_coords = new_coords[0].numpy()
util.cond_mkdir(results_dir)
np.save(os.path.join(results_dir, 'coords.npy'), raw_coords)
np.save(os.path.join(results_dir, 'new_coords.npy'), new_coords)

moving_img = nrrd.read(os.path.join(opt.image_dir, opt.moving_pid, 'img.nrrd'))[0]
moving_mask = util.load_mask(opt.mask_dir, opt.moving_pid, roi_names)

warped_moving_img = util.warp_3d_volume(moving_img, new_coords, ct_dataset.maxlen, 3)
nrrd.write(os.path.join(results_dir, 'warped_img.nrrd'), warped_moving_img)
for i, roi_name in enumerate(roi_names):
    warped_moving_mask = util.warp_3d_volume(moving_mask[i], new_coords, ct_dataset.maxlen, 0)
    nrrd.write(os.path.join(results_dir, f'warped_{roi_name}.nrrd'), warped_moving_mask)

print(f"Finish {opt.moving_pid}: {opt.fixed_pid}")

