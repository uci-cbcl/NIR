# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from src import dataio, training, loss_functions, modules, efficient_modules, deform_net
from src.util import write_3d_img_pairwise_reg_summary

import torch
from torch.utils.data import DataLoader
import configargparse
from functools import partial

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--reg_logging_root', type=str, default='./logs_HLN-12-1', help='image registration model logging root dir')
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

p.add_argument('--sample_mtd', type=str, default="random", help='Sampling methods')
p.add_argument('--sample_exp', type=float, default=18, help='Number of sampling points is 2**sampling_exp')
p.add_argument('--gaussian_scale', type=float, default=5, help='Gaussian sclae for position encoding')
p.add_argument('--num_hidden_layers', type=int, default=3, help='Number of hidden layers in the mlps')
p.add_argument('--loss_type', type=str, default='image_mse', help='Loss Function to minimize')
p.add_argument('--image_dir', type=str, help='3D Medical fixed Image dir')
p.add_argument('--fixed_pid', type=str, help='3D Medical fixed Image patient id')
p.add_argument('--moving_pid', type=str, help='3D Medical moving Image patient id')
opt = p.parse_args()

# torch.set_default_dtype(torch.float16)

print(f'Moving: {opt.moving_pid}; Fixed: {opt.fixed_pid}')

ct_dataset = dataio.Single_CT_Brain(os.path.join(opt.image_dir, opt.fixed_pid), maxlen=None)
coord_dataset = dataio.Implicit3DWrapper(ct_dataset, sample_method=opt.sample_mtd, sample_exp=opt.sample_exp)
dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)
ct_dataset.maxlen = coord_dataset.maxlen

moving_data = dataio.Single_CT_Brain(os.path.join(opt.image_dir, opt.moving_pid), maxlen=None)
fixed_data = dataio.Single_CT_Brain(os.path.join(opt.image_dir, opt.fixed_pid), maxlen=None)

moving_model = dataio.Single_CT_Brain_Model(torch.from_numpy(moving_data.v).float().cuda(), ct_dataset.shape, ct_dataset.maxlen)
fixed_model = dataio.Single_CT_Brain_Model(torch.from_numpy(fixed_data.v).float().cuda(), ct_dataset.shape, ct_dataset.maxlen)
moving_v = torch.from_numpy(moving_data.v).float().cuda()    

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
    deform_model = deform_net.DiffeoDeformedField(deform_field_model, moving_v)

else:
    deform_field_model = efficient_modules.EfficientMLP(in_features=3, out_features=len(ct_dataset.shape), num_hidden_layers=opt.num_hidden_layers,
                                                        mode=opt.model_type.split('_')[0], gaussian_scale=opt.gaussian_scale, nonlinearity=opt.model_type.split('_')[-1])
    
    deform_model = deform_net.DeformedField(deform_field_model, 'residual', moving_v)

deform_model.cuda()
deform_model.train()

# Define the loss
loss_types = opt.loss_type.split('+')
loss_weight = float(opt.experiment_name.split('_')[-1])

root_path = os.path.join(opt.reg_logging_root, 'reg', f'from_{opt.moving_pid}_to_{opt.fixed_pid}', f'{opt.experiment_name}_{opt.loss_type}')

summary_fn = partial(write_3d_img_pairwise_reg_summary, ct_dataset)

training.train_efficient_pairwise(deform_model, moving_model, fixed_model,
                        train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                        steps_til_summary=opt.steps_til_summary//opt.batch_size, epochs_til_checkpoint=opt.epochs_til_ckpt,
                        model_dir=root_path, summary_fn=summary_fn, loss_types=loss_types,
                        loss_weight=loss_weight, maxlen=coord_dataset.maxlen)

