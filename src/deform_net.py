# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Define DIF-Net
'''

from operator import mod
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from loss_functions import image_mse
from torchdiffeq import odeint_adjoint as odeint

from einops import rearrange

class GridField(nn.Module):
    def __init__(self, grid_deform):
        super().__init__()
        # Deform-Net
        self.grid_deform = torch.nn.parameter.Parameter(grid_deform, requires_grad=True)

    def forward(self):
        return self.grid_deform


class DeformedGridField(nn.Module):
    def __init__(self, grid_deform, model_type='residual', moving_img=None, **kwargs):
        super().__init__()
        # Deform-Net
        self.deform_model = GridField(grid_deform)
        self.model_type = model_type
        self.moving_img = moving_img

    # for generation
    def deform(self, coords):

        with torch.no_grad():

            if self.model_type == 'residual':
                deformation = self.deform_model()
                new_coords = coords + deformation
            elif self.model_type == 'dst':
                new_coords = self.deform_model()

            return new_coords

    # run moving image model
    def get_moving_image(self, coords):
        model_in = {'coords': coords}
        return self.moving_img_model(model_in)['model_out']

    # for training
    def forward(self, model_input, moving_img_model, **kwargs):
        # [deformation field, correction field]
        model_in = model_input['coords']

        if self.model_type == 'residual':
            deformation = self.deform_model()
            new_coords = model_in + deformation
        elif self.model_type == 'dst':
            new_coords = self.deform_model()

        if self.moving_img is not None:
            dims_num = new_coords.dim()
            if  dims_num == 6:
                B, X, Y, Z, num_chunks, channel = new_coords.shape
                new_coords = new_coords.reshape(B, X, Y, -1, channel)
            reg_output_v = (F.grid_sample(self.moving_img[None, None], 
                                          new_coords[..., [2,1,0]], 
                                          mode='bilinear', 
                                          padding_mode='zeros').permute((0,2,3,4,1)) - 0.5) * 2
            if dims_num == 6:
                new_coords = new_coords.reshape(B, X, Y, Z, num_chunks, channel)
                reg_output_v = reg_output_v.reshape(B, X, Y, Z, num_chunks)
            reg_output = {'model_out': reg_output_v}
        else:
            model_input_temp = {'coords':new_coords}
            reg_output = moving_img_model(model_input_temp)

        reg_output['new_coords'] = new_coords
        reg_output['model_in'] = model_in

        return reg_output


class DeformedField(nn.Module):
    def __init__(self, deform_model, model_type='residual', moving_img=None, **kwargs):
        super().__init__()
        # Deform-Net
        self.deform_model = deform_model
        self.model_type = model_type
        self.moving_img = moving_img

    # for generation
    def deform(self, coords):

        with torch.no_grad():
            model_in = {'coords': coords}
            model_output = self.deform_model(model_in)

            if self.model_type == 'residual':
                deformation = model_output['model_out'][:,:,:3]
                new_coords = coords + deformation
            elif self.model_type == 'dst':
                new_coords = model_output['model_out'][:,:,:3]

            return new_coords

    # run moving image model
    def get_moving_image(self, coords):
        model_in = {'coords': coords}
        return self.moving_img_model(model_in)['model_out']

    # for training
    def forward(self, model_input, moving_img_model, **kwargs):
        # [deformation field, correction field]
        model_output = self.deform_model(model_input)

        if self.model_type == 'residual':
            deformation = model_output['model_out'][...,:3]
            new_coords = model_output['model_in'] + deformation
        elif self.model_type == 'dst':
            new_coords = model_output['model_out'][...,:3]

        if self.moving_img is not None:
            dims_num = new_coords.dim()
            if  dims_num == 6:
                B, X, Y, Z, num_chunks, channel = new_coords.shape
                new_coords = new_coords.reshape(B, X, Y, -1, channel)
            stacked_new_coords = rearrange(new_coords, 'b h w d c -> 1 (b h) w d c')
            reg_output_v = (F.grid_sample(self.moving_img[None, None], 
                                          stacked_new_coords[..., [2,1,0]], 
                                          mode='bilinear', 
                                          padding_mode='zeros').permute((0,2,3,4,1)) - 0.5) * 2
            reg_output_v = rearrange(reg_output_v, '1 (b h) w d 1 -> b h w d 1', b=len(new_coords))
            if dims_num == 6:
                new_coords = new_coords.reshape(B, X, Y, Z, num_chunks, channel)
                reg_output_v = reg_output_v.reshape(B, X, Y, Z, num_chunks)
            reg_output = {'model_out': reg_output_v}
        else:
            model_input_temp = {'coords':new_coords}
            reg_output = moving_img_model(model_input_temp)

        reg_output['new_coords'] = new_coords
        reg_output['model_in'] = model_output['model_in']

        return reg_output


class ODEFunc(nn.Module):
    '''
    This refers to the dynamics function f(x,t) in a IVP defined as dh(x,t)/dt = f(x,t). 
    For a given location (t) on point (x) trajectory, it returns the direction of 'flow'.
    Refer to Section 3 (Dynamics Equation) in the paper for details. 
    '''
    def __init__(self, positional_encoding, dynamic_net):
        '''
        Initialization. 
        num_hidden: number of nodes in a hidden layer
        latent_len: size of the latent code being used
        '''
        
        super(ODEFunc, self).__init__()
        
        self.positional_encoding = positional_encoding
        self.dynamic_net = dynamic_net 
        self.tanh = nn.Tanh()
        
        self.nfe = 0
        
    def forward(self, t, xyz):    
        if self.positional_encoding is None:
            xyz_emb = xyz
        else:
            xyz_emb = self.positional_encoding(xyz)
        xyz_deform = self.tanh(self.dynamic_net(xyz_emb)) #Computed dynamics of point x at time t

        self.nfe+=1  #To check #ode evaluations
        return xyz_deform # output is therefore like [0,0..,0, dyn_x, dyn_y, dyn_z] for a point


class NODEBlock(nn.Module):
    '''
    Function to solve an IVP defined as dh(x,t)/dt = f(x,t). 
    We use the differentiable ODE Solver by Chen et.al used in their NeuralODE paper.
    '''
    def __init__(self, odefunc):
        '''
        Initialization. 
        odefunc: The dynamics function to be used for solving IVP
        tol: tolerance of the ODESolver
        '''
        super(NODEBlock, self).__init__()
        self.odefunc = odefunc
        self.cost = 0
        
    def forward(self, model_input, step_size=0.05):
        '''
        Solves the ODE in the forward / reverse time. 
        '''
        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords']
        coords_shape = coords_org.shape
        coords = coords_org.reshape(-1, coords_shape[-1])
        self.odefunc.nfe = 0  #To check #ode evaluations

        # Solve the ODE with initial condition x and interval time.
        times = torch.tensor([0, 0.2]).to(coords)

        out = odeint(self.odefunc, coords, times, rtol=1e-3, atol=1e-5, 
                     method='rk4', options={'step_size': step_size})[-1]
        # , method='rk4', options={'step_size': step_size}
        # out = self.odefunc(None, coords)
        self.cost = self.odefunc.nfe  # Number of evaluations it took to solve it
        deformation = out.reshape(coords_shape)
        return {'model_in': coords_org, 'model_out': deformation}
        # return {'model_in': coords_org, 'model_out': coords}


class DiffeoDeformedField(nn.Module):
    def __init__(self, ode_func, moving_img=None, **kwargs):
        super().__init__()
        # Deform-Net
        self.deform_model = NODEBlock(ode_func)
        print(self.deform_model)
        
        self.moving_img = moving_img

    def deform(self, coords):
        with torch.no_grad():
            model_in = {'coords': coords}
            new_coords = self.deform_model(model_in)['model_out'][...,:3]

            return new_coords

    # for training
    def forward(self, model_input, moving_img_model, **kwargs):
        # [deformation field, correction field]
        model_output = self.deform_model(model_input)

        new_coords = model_output['model_out'][...,:3]

        if self.moving_img is not None:
            dims_num = new_coords.dim()
            if  dims_num == 6:
                B, X, Y, Z, num_chunks, channel = new_coords.shape
                new_coords = new_coords.reshape(B, X, Y, -1, channel)
            stacked_new_coords = rearrange(new_coords, 'b h w d c -> 1 (b h) w d c')
            reg_output_v = (F.grid_sample(self.moving_img[None, None], 
                                          stacked_new_coords[..., [2,1,0]], 
                                          mode='bilinear', 
                                          padding_mode='zeros').permute((0,2,3,4,1)) - 0.5) * 2
            reg_output_v = rearrange(reg_output_v, '1 (b h) w d 1 -> b h w d 1', b=len(new_coords))
            if dims_num == 6:
                new_coords = new_coords.reshape(B, X, Y, Z, num_chunks, channel)
                reg_output_v = reg_output_v.reshape(B, X, Y, Z, num_chunks)
            reg_output = {'model_out': reg_output_v}
        else:
            model_input_temp = {'coords':new_coords}
            reg_output = moving_img_model(model_input_temp)

        reg_output['new_coords'] = new_coords
        reg_output['model_in'] = model_output['model_in']

        return reg_output
    

class DeformedMSSField(nn.Module):
    def __init__(self, deform_model_0, deform_model_1, model_type='residual', moving_img=None, **kwargs):
        super().__init__()
        # Deform-Net
        self.deform_model_0 = deform_model_0
        self.deform_model_1 = deform_model_1
        self.model_type = model_type
        self.moving_img = moving_img

    # for generation
    def deform(self, coords):
        with torch.no_grad():
            model_in = {'coords': coords}
            model_output_0 = self.deform_model_0(model_in)

            if self.model_type == 'residual':
                deformation = model_output_0['model_out'][:,:,:3]
                new_coords_0 = coords + deformation
            elif self.model_type == 'final':
                new_coords_0 = model_output_0['model_out'][:,:,:3]

            model_in_1 = {'coords': new_coords_0}
            model_output_1 = self.deform_model_1(model_in_1)

            if self.model_type == 'residual':
                deformation = model_output_1['model_out'][:,:,:3]
                new_coords = new_coords_0 + deformation
            elif self.model_type == 'final':
                new_coords = model_output_1['model_out'][:,:,:3]

            return new_coords

    # run moving image model
    def get_moving_image(self, coords):
        model_in = {'coords': coords}
        return self.moving_img_model(model_in)['model_out']

    # for training
    def training_forward(self, model_input_1, moving_img_model, **kwargs):
        # [deformation field, correction field]
        with torch.no_grad():
            model_output_1_initial = self.deform_model_0(model_input_1)

        if self.model_type == 'residual':
            deformation = model_output_1_initial['model_out'][...,:3]
            new_coords_1_initial = model_output_1_initial['model_in'] + deformation
        elif self.model_type == 'dst':
            new_coords_1_initial = model_output_1_initial['model_out'][...,:3]

        model_input_1_initial = {'coords': new_coords_1_initial}
        model_output_1 = self.deform_model_1(model_input_1_initial)

        if self.model_type == 'residual':
            deformation = model_output_1['model_out'][...,:3]
            new_coords_1 = new_coords_1_initial + deformation
        elif self.model_type == 'dst':
            new_coords_1 = model_output_1['model_out'][...,:3]

        if self.moving_img is not None:
            dims_num = new_coords_1.dim()
            if  dims_num == 6:
                B, X, Y, Z, num_chunks, channel = new_coords_1.shape
                new_coords_1 = new_coords_1.reshape(B, X, Y, -1, channel)
            stacked_new_coords_1 = rearrange(new_coords_1, 'b h w d c -> 1 (b h) w d c')
            reg_output_v_1 = (F.grid_sample(self.moving_img[None, None], 
                                          stacked_new_coords_1[..., [2,1,0]], 
                                          mode='bilinear', 
                                          padding_mode='zeros').permute((0,2,3,4,1)) - 0.5) * 2
            reg_output_v_1 = rearrange(reg_output_v_1, '1 (b h) w d 1 -> b h w d 1', b=len(new_coords_1))
            if dims_num == 6:
                new_coords_1 = new_coords_1.reshape(B, X, Y, Z, num_chunks, channel)
                reg_output_v_1 = reg_output_v_1.reshape(B, X, Y, Z, num_chunks)
            reg_output_1 = {'model_out': reg_output_v_1}
        else:
            model_input_temp = {'coords':new_coords_1}
            reg_output_1 = moving_img_model(model_input_temp)

        reg_output_1['new_coords'] = new_coords_1
        reg_output_1['model_in'] = model_output_1['model_in']
        
        return reg_output_1


    # for inference
    def forward(self, model_input, moving_img_model, **kwargs):
        # [deformation field, correction field]
        model_output_0 = self.deform_model_0(model_input)

        if self.model_type == 'residual':
            deformation = model_output_0['model_out'][...,:3]
            new_coords_0 = model_output_0['model_in'] + deformation
        elif self.model_type == 'dst':
            new_coords_0 = model_output_0['model_out'][...,:3]

        model_input_1 = {'coords': new_coords_0}
        model_output_1 = self.deform_model_1(model_input_1)

        if self.model_type == 'residual':
            deformation = model_output_1['model_out'][...,:3]
            new_coords_1 = new_coords_0 + deformation
        elif self.model_type == 'dst':
            new_coords_1 = model_output_1['model_out'][...,:3]

        if self.moving_img is not None:
            dims_num = new_coords_1.dim()
            if  dims_num == 6:
                B, X, Y, Z, num_chunks, channel = new_coords_1.shape
                new_coords_1 = new_coords_1.reshape(B, X, Y, -1, channel)
            stacked_new_coords_1 = rearrange(new_coords_1, 'b h w d c -> 1 (b h) w d c')
            reg_output_v = (F.grid_sample(self.moving_img[None, None], 
                                          stacked_new_coords_1[..., [2,1,0]], 
                                          mode='bilinear', 
                                          padding_mode='zeros').permute((0,2,3,4,1)) - 0.5) * 2
            reg_output_v = rearrange(reg_output_v, '1 (b h) w d 1 -> b h w d 1', b=len(new_coords_1))
            if dims_num == 6:
                new_coords_1 = new_coords_1.reshape(B, X, Y, Z, num_chunks, channel)
                reg_output_v = reg_output_v.reshape(B, X, Y, Z, num_chunks)
            reg_output = {'model_out': reg_output_v}
        else:
            model_input_temp = {'coords':new_coords_1}
            reg_output = moving_img_model(model_input_temp)

        reg_output['new_coords'] = new_coords_1
        reg_output['model_in'] = model_output_1['model_in']

        return reg_output


class DiffeoDeformedMSSField(nn.Module):
    def __init__(self, ode_func_0, ode_func_1, moving_img=None, **kwargs):
        super().__init__()
        # Deform-Net
        self.deform_model_0 = NODEBlock(ode_func_0)
        self.deform_model_1 = NODEBlock(ode_func_1)
        
        self.moving_img = moving_img

    def deform(self, coords):
        with torch.no_grad():
            model_in_0 = {'coords': coords}
            new_coords_0 = self.deform_model_0(model_in_0)['model_out'][...,:3]
            model_in_1 = {'coords': new_coords_0}
            new_coords_1 = self.deform_model_1(model_in_1)['model_out'][...,:3]

            return new_coords_1

    # for training
    def training_forward(self, model_input_1, moving_img_model, **kwargs):
        with torch.no_grad():
            model_output_1_initial = self.deform_model_0(model_input_1)
        model_input_1_initial = {'coords': model_output_1_initial['model_out'][...,:3]}
        model_output_1 = self.deform_model_1(model_input_1_initial)
        new_coords_1 = model_output_1['model_out'][...,:3]

        if self.moving_img is not None:
            dims_num = new_coords_1.dim()
            if  dims_num == 6:
                B, X, Y, Z, num_chunks, channel = new_coords_1.shape
                new_coords_1 = new_coords_1.reshape(B, X, Y, -1, channel)
            stacked_new_coords_1 = rearrange(new_coords_1, 'b h w d c -> 1 (b h) w d c')
            reg_output_v_1 = (F.grid_sample(self.moving_img[None, None], 
                                          stacked_new_coords_1[..., [2,1,0]], 
                                          mode='bilinear', 
                                          padding_mode='zeros').permute((0,2,3,4,1)) - 0.5) * 2
            reg_output_v_1 = rearrange(reg_output_v_1, '1 (b h) w d 1 -> b h w d 1', b=len(new_coords_1))
            if dims_num == 6:
                new_coords_1 = new_coords_1.reshape(B, X, Y, Z, num_chunks, channel)
                reg_output_v_1 = reg_output_v_1.reshape(B, X, Y, Z, num_chunks)
            reg_output_1 = {'model_out': reg_output_v_1}
        else:
            dims_num = new_coords_1.dim()
            if  dims_num == 6:
                B, X, Y, Z, num_chunks, channel = new_coords_1.shape
                new_coords_1 = new_coords_1.reshape(B, X, Y, -1, channel)
            model_input_temp = {'coords':new_coords_1}
            reg_output_1 = moving_img_model(model_input_temp)
            if dims_num == 6:
                new_coords_1 = new_coords_1.reshape(B, X, Y, Z, num_chunks, channel)
                reg_output_1['model_out'] = reg_output_1['model_out'].reshape(B, X, Y, Z, num_chunks)

        reg_output_1['new_coords'] = new_coords_1
        reg_output_1['model_in'] = model_output_1['model_in']
        
        return reg_output_1

    # for infernce
    def forward(self, model_input, moving_img_model, **kwargs):
        # [deformation field, correction field]
        model_output_0 = self.deform_model_0(model_input)
        new_coords_0 = model_output_0['model_out'][...,:3]
        model_input_1 = {'coords': new_coords_0}
        model_output_1 = self.deform_model_1(model_input_1)
        new_coords_1 = model_output_1['model_out'][...,:3]

        if self.moving_img is not None:
            stacked_new_coords_1 = rearrange(new_coords_1, 'b h w d c -> 1 (b h) w d c')
            reg_output_v = (F.grid_sample(self.moving_img[None, None], 
                                          stacked_new_coords_1[..., [2,1,0]], 
                                          mode='bilinear', 
                                          padding_mode='zeros').permute((0,2,3,4,1)) - 0.5) * 2
            reg_output_v = rearrange(reg_output_v, '1 (b h) w d 1 -> b h w d 1', b=len(new_coords_1))
            reg_output = {'model_out': reg_output_v}
        else:
            model_input_temp = {'coords':new_coords_1}
            reg_output = moving_img_model(model_input_temp)

        reg_output['new_coords'] = new_coords_1
        reg_output['model_in'] = model_output_0['model_in']

        return reg_output
