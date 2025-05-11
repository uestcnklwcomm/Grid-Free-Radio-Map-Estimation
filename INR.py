import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from collections import OrderedDict

from scipy.io import loadmat, savemat

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import torchvision.transforms.functional as Funvision
import numpy as np
from NMF import tensor_unfold_m3
import skimage

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=10):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=10, hidden_omega_0=10):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        """Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!"""
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def get_target_tensor(img, sidelength):
    transform = Compose([
        Resize(sidelength, interpolation=Funvision.InterpolationMode.NEAREST),
        # Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])

    img = transform(img)
    return img


class TargetFitting(Dataset):
    def __init__(self, image, sidelength):
        super().__init__()
        img = get_target_tensor(image, sidelength)
        self.values = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.values


def large_scale_tensor_downsamp(tensor, sidelength):
    transform = Compose([
        Resize(sidelength, interpolation=Funvision.InterpolationMode.NEAREST)])
    i, j, k = tensor.shape
    tensor_resize = np.zeros([sidelength, sidelength, k])
    for kk in range(0, k):
        tensor_slab = np.squeeze(tensor[:, :, kk])
        tensor_slab_to_image = Image.fromarray(tensor_slab)
        slab_transform = np.array(transform(tensor_slab_to_image))
        tensor_resize[:, :, kk] = slab_transform

    return tensor_resize


def fibersampling(tensor, samplingsize):
    tensor_mode3 = tensor_unfold_m3(tensor)
    fibernum = tensor_mode3.shape[0]
    sampled_indices = np.random.choice(fibernum, size=samplingsize, replace=False)
    fibers = tensor_mode3[sampled_indices]

    return fibers, sampled_indices


def fibersampling_exclude(tensor, samplingsize, exclude_idx):
    tensor_mode3 = tensor_unfold_m3(tensor)
    fibernum = tensor_mode3.shape[0]
    indices_set = np.linspace(0, fibernum-1, num=fibernum)
    boolmask = np.ones(len(indices_set), dtype=bool)
    boolmask[exclude_idx] = False
    indices_set = indices_set[boolmask]

    sel_indices = np.random.choice(len(indices_set), size=samplingsize, replace=False)
    sampled_indices = np.array(indices_set[sel_indices], dtype=int)

    fibers = tensor_mode3[sampled_indices]

    return fibers, sampled_indices


class SLFFitting_Regular(Dataset):
    def __init__(self, slfvec, sidelength):
        super().__init__()
        slf_to_tensor = torch.from_numpy(slfvec)
        self.values = slf_to_tensor.unsqueeze(1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.values


class SLFFitting_Irregular(Dataset):
    def __init__(self, slfvec, fullsize, samplingindex):
        super().__init__()
        slf_to_tensor = torch.from_numpy(slfvec)
        mgrids = get_mgrid(fullsize, 2)

        if isinstance(samplingindex, np.ndarray):
            samplingindex = torch.from_numpy(samplingindex)

        if (samplingindex < 0).any() or (samplingindex >= fullsize ** 2).any():
            raise ValueError("Sampling index invalid")

        downsampled_grids = mgrids[samplingindex, :]

        self.values = slf_to_tensor.unsqueeze(1)
        self.coords = downsampled_grids

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.values
