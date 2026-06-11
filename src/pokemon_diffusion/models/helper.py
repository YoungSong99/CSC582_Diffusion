import torch.nn as nn

def get_activation(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in [None, "none"]:
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")

def get_output_activation(name):
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name in [None, "none"]:
        return nn.Identity()
    raise ValueError(f"Unknown output activation: {name}")

def get_norm_layer(norm_type, num_channels, img_size=None):
    if norm_type in [None, "none"]:
        return nn.Identity()
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm_type == "group":
        groups = min(32, num_channels)
        return nn.GroupNorm(groups, num_channels)
    if norm_type == "layer":
        return nn.GroupNorm(1, num_channels)
    raise ValueError(f"Unknown norm_type: {norm_type}")