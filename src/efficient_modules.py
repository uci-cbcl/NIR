import tinycudann as tcnn
import torch
from torch import nn
from torch.nn import Module, Sequential, Linear
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F

import modules


class EfficientMLP(Module):

    def __init__(self, out_features=1, in_features=2, mode='gaussian', hidden_features=256, num_hidden_layers=3, 
                 nonlinearity='sine', zero_weight_init_outmost_linear=False, **kwargs):
        super().__init__()
        self.mode = mode

        if mode == 'gaussian':
            self.positional_encoding = PosEncodingFourier(in_features=in_features, gaussian_scale=kwargs.get('gaussian_scale', 1), embedding_size=128)
            n_input_dims = self.positional_encoding.out_dim
        elif mode == 'hash':
            max_res = 128
            min_res = 16
            num_levels = 8
            n_features_per_level = 8
            self.positional_encoding = HashEncoding(in_features, max_res, min_res, num_levels, n_features_per_level=n_features_per_level, log2_hashmap_size=16)
            n_input_dims = n_features_per_level * num_levels
        
        self.net = modules.FCBlock(in_features=n_input_dims, out_features=out_features, num_hidden_layers=num_hidden_layers,
                                    hidden_features=hidden_features, outermost_linear=True, 
                                    nonlinearity=nonlinearity, zero_weight_init_outmost_linear=zero_weight_init_outmost_linear)
        # network_config = {
        #     "otype": "CutlassMLP",       
        #     "activation": "ReLU",        
        #     "output_activation": "None", 
        #     "n_neurons": 256,            
        #     "n_hidden_layers": 4        
        # }
        # self.net = tcnn.Network(n_input_dims, out_features, network_config)

        print(self)

    def forward(self, model_input):

        # Enables us to compute gradients w.r.t. coordinates
        # coords_org = model_input['coords'].requires_grad_(True)
        # coords = coords_org
        coords_org = model_input['coords']
        coords_shape = coords_org.shape
        coords_org = coords_org.reshape(-1, coords_shape[-1])

        # various input processing methods for different applications
        if self.mode == 'gaussian':
            coords_emb = self.positional_encoding(coords_org)
        elif self.mode == 'hash':
            coords_emb = self.positional_encoding(coords_org)

        # import pdb; pdb.set_trace()
        output = self.net(coords_emb)

        coords_org = coords_org.reshape(coords_shape)
        output = output.reshape(coords_shape)

        return {'model_in': coords_org, 'model_out': output}


class HashEncoding(nn.Module):
    def __init__(self, in_features, max_res, min_res, num_levels, n_features_per_level=2, log2_hashmap_size=16, dtype=torch.float, **kwargs):
        super().__init__()

        growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1)) if num_levels > 1 else 1
        encoding_config = {
            "otype": "Grid",          
            "type": "Hash",  
            "n_levels": num_levels,  
            "n_features_per_level": n_features_per_level, 
            "log2_hashmap_size": log2_hashmap_size, 
            "base_resolution": min_res,   
            "per_level_scale": growth_factor,   
            "interpolation": "Linear" 
        }
        self.hash_encoding = tcnn.Encoding(in_features, encoding_config, dtype=dtype)

    def forward(self, coords):
        coords = (coords + 1) / 2
        return self.hash_encoding(coords)



class PosEncodingFourier(nn.Module):
    def __init__(self, in_features, **kwargs):
        super().__init__()
        gaussian_scale = kwargs.get('gaussian_scale', 5.)
        embedding_size = kwargs.get('embedding_size', 256)
        bvals = torch.randn((embedding_size, 3)) * gaussian_scale
        avals = torch.ones(bvals.shape[0])
        self.out_dim = embedding_size * 2

        self.register_buffer('bvals', bvals)
        self.register_buffer('avals', avals)

    def forward(self, coords):
        self.avals = self.avals.to(coords)
        self.bvals = self.bvals.to(coords)
        sins = self.avals * torch.sin(torch.matmul(np.pi * coords, self.bvals.T))
        coss = self.avals * torch.cos(torch.matmul(np.pi * coords, self.bvals.T))
        return torch.cat([sins, coss], axis=-1)

