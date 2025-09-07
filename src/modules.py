import torch
from torch import nn
from torch.nn import Module, Sequential, Linear
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        # return torch.sin(30 * input)
        return torch.sin(10 * input)


class FCBlock(Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', 
                 weight_init=None, zero_weight_init_outmost_linear=False):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(Sequential(
            Linear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(Sequential(
                Linear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(Sequential(Linear(hidden_features, out_features)
            ))
        else:
            self.net.append(Sequential(
                Linear(hidden_features, out_features), nl
            ))

        self.net = Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

        if zero_weight_init_outmost_linear:
            self.net[-1].apply(init_weights_zeros)

    def forward(self, coords, **kwargs):

        output = self.net(coords)
        return output


class SingleBVPNet(Module):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode

        if self.mode in ['basic', 'posenc', 'posenc_new']:
            self.fourier_positional_encoding = PosEncodingFourier(in_features=in_features, method=self.mode)
            in_features = self.fourier_positional_encoding.out_dim
        elif self.mode == 'gaussian':
            self.fourier_positional_encoding = PosEncodingFourier(in_features=in_features, method=self.mode,
                                                                  gaussian_scale=kwargs.get('gaussian_scale', 1), embedding_size=128)
            in_features = self.fourier_positional_encoding.out_dim
        
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                            hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        print(self)

    def forward(self, model_input):

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords']
        coords = coords_org

        # various input processing methods for different applications
        if self.mode in ['basic', 'posenc', 'posenc_new', 'gaussian']:
            coords_emb = self.fourier_positional_encoding(coords_org)
        else:
            coords_emb = coords_org

        output = self.net(coords_emb)

        return {'model_in': coords_org, 'model_out': output}

    def chunk_forward(self, model_input):
        # with torch.no_grad():
        coords_all = model_input['coords']
        slice_num = coords_all.shape[1]

        coords_chunk = torch.chunk(coords_all, slice_num, 1)

        whole_output = []
        for coords in coords_chunk:
            model_output = self.forward({'coords': coords})['model_out']
            whole_output += [model_output]

        whole_output = torch.cat(whole_output, 1)

        return {'model_out': whole_output}


class PosEncodingFourier(nn.Module):
    def __init__(self, in_features, method='basic', **kwargs):
        super().__init__()
        if method == 'basic':
            bvals = torch.eye(3)
            avals = torch.ones((bvals.shape[0]))
            self.out_dim = in_features * 2
        if method == 'posenc':
            max_posenc_log_scale = kwargs.get('max_posenc_log_scale', 8)
            bvals = 2. ** (torch.arange(0, max_posenc_log_scale))
            bvals = torch.reshape(torch.eye(3)*bvals[:,None,None], [len(bvals)*3, 3])
            avals = torch.ones(bvals.shape[0])
            self.out_dim = bvals.shape[0] * 2
        if method == 'posenc_new':
            max_posenc_log_scale = kwargs.get('max_posenc_log_scale', 8)
            embedding_size = kwargs.get('embedding_size', 256)
            bvals = 2. ** torch.linspace(0, max_posenc_log_scale, embedding_size//3)
            bvals = torch.reshape(torch.eye(3)*bvals[:,None,None], [len(bvals)*3, 3])
            avals = torch.ones(bvals.shape[0])
            self.out_dim = bvals.shape[0] * 2
        if method == 'gaussian':
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


########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def init_weights_zeros(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            # m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def init_recurrent_weights(self):
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)


def lstm_forget_gate_init(lstm_layer):
    for name, parameter in lstm_layer.named_parameters():
        if not "bias" in name: continue
        n = parameter.size(0)
        start, end = n // 4, n // 2
        parameter.data[start:end].fill_(1.)


def clip_grad_norm_hook(x, max_norm=10):
    total_norm = x.norm()
    total_norm = total_norm ** (1 / 2.)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        return x * clip_coef


def init_out_weights(self):
    for m in self.modules():
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -1e-5, 1e-5)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

