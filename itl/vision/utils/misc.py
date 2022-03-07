"""
Miscellaneous utility methods that don't classify into other files in utils
"""
import re
import requests

import torch
from tqdm import tqdm
from detectron2.checkpoint.c2_model_loading import convert_basic_c2_names


def pair_vals(vals):
    """
    Generate possible value pairs given tensor of values, except diagonals where i==j,
    treating each first dim entry as value
    """
    n = len(vals)
    grid_size = [n,n] + list(vals.shape[1:])

    grid = torch.stack([vals[:, None].expand(grid_size), vals[None, :].expand(grid_size)], dim=2)
    grid = grid.view([n**2, 2] + list(vals.shape[1:]))

    inds_diag = -torch.arange(n-1,-1,-1) * (n+1) + n**2 - 1
    inds_of_inds = torch.ones(n**2, dtype=torch.bool)
    inds_of_inds[inds_diag] = False

    return grid[inds_of_inds]

def pair_select_indices(N, inds):
    """
    Generate indices for selecting rows corresponding to matching pairs, from tensor
    which is indexed by all possible pairs of indices except diagonals where i==j 
    (generally expected to be outputs from pair_vals() method above)
    """
    if len(inds.shape) == 0: inds = inds[None]    # Unsqueeze singleton tensors

    ind_pairs = pair_vals(inds)

    assert len(ind_pairs.shape) == 2 and ind_pairs.shape[-1] == 2

    inds1 = ind_pairs[:, 0]; inds2 = ind_pairs[:, 1]
    
    return inds1 * (N-1) + inds2 + (inds1 > inds2) - 1

def download_from_url(url, target_path):
    """
    Download a single file from the given url to the target path, with a progress bar shown
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))

    with open(target_path, 'wb') as file, tqdm(
        desc=target_path.split("/")[-1],
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def has_model(func):
    """Short decorator for asserting self's model is not None"""
    def wrapper(self, *args, **kwargs):
        assert hasattr(self, "model") and self.model is not None
        return func(self, *args, **kwargs)

    return wrapper

def rename_resnet_params(params):
    """
    Rename parameters in state_dict of pre-trained ResNet weights from torch hub
    """
    ## Handling weight names from detectron2 repo (Caffe2 naming conv)
    old_pnames = sorted(list(params.keys()))
    new_pnames = convert_basic_c2_names(old_pnames)

    for old_pname, new_pname in zip(old_pnames, new_pnames):
        # Convert the numpy params to torch tensors along the way as well
        params[f"bottom_up.{new_pname}"] = torch.tensor(params.pop(old_pname))

    ## Handling weight names from torch hub
    stem_conv1_pattern = "^conv1\.weight$"
    stem_bn1_pattern = "^bn1\.(\w+)$"
    layer_conv_pattern = "^layer(\d)\.(\d)\.conv(\d)\.weight$"
    layer_bn_pattern = "^layer(\d)\.(\d)\.bn(\d)\.(\w+)$"
    layer_ds0_pattern = "^layer(\d)\.(\d)\.downsample\.0\.weight$"
    layer_ds1_pattern = "^layer(\d)\.(\d)\.downsample\.1\.(\w+)$"

    prefix = "bottom_up"
    old_pnames = list(params.keys())

    for old_pname in old_pnames:
        new_pname = ""

        # conv1.weight => stem.conv1.weight
        match = re.match(stem_conv1_pattern, old_pname)
        if match is not None:
            m = match.groups()
            new_pname = f"stem.conv1.weight"

        # E.g. bn1.{param} => stem.conv1.norm.{param}
        match = re.match(stem_bn1_pattern, old_pname)
        if match is not None:
            m = match.groups()
            new_pname = f"stem.conv1.norm.{m[0]}"

        # E.g. layer1.0.conv1.weight => res2.0.conv1.weight
        match = re.match(layer_conv_pattern, old_pname)
        if match is not None:
            m = match.groups()
            new_pname = f"res{int(m[0])+1}.{m[1]}.conv{m[2]}.weight"
        
        # E.g. layer1.0.bn1.{param} => res2.0.conv1.norm.{param}
        match = re.match(layer_bn_pattern, old_pname)
        if match is not None:
            m = match.groups()
            new_pname = f"res{int(m[0])+1}.{m[1]}.conv{m[2]}.norm.{m[3]}"
        
        # E.g. layer1.0.downsample.0.weight => res2.0.shortcut.weight
        match = re.match(layer_ds0_pattern, old_pname)
        if match is not None:
            m = match.groups()
            new_pname = f"res{int(m[0])+1}.{m[1]}.shortcut.weight"

        # E.g. layer1.0.downsample.1.{param} => res2.0.shortcut.norm.{param}
        match = re.match(layer_ds1_pattern, old_pname)
        if match is not None:
            m = match.groups()
            new_pname = f"res{int(m[0])+1}.{m[1]}.shortcut.norm.{m[2]}"

        # Change key by popping-setting
        if new_pname:
            params[f"{prefix}.{new_pname}"] = params.pop(old_pname)
