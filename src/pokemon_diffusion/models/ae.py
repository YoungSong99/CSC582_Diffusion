import torch
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


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation="relu", norm_type="none"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            get_norm_layer(norm_type, out_ch),
            get_activation(activation),
        )

    def forward(self, x):
        return self.block(x)
    

class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation="relu", norm_type="none"):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            get_norm_layer(norm_type, out_ch),
            get_activation(activation),
        )

    def forward(self, x):
        return self.block(x)


class AutoEncoder(nn.Module):
    def __init__(self, img_size=96, latent_dim=128, activation="relu", norm_type="none", output_activation="tanh", base_channels=32,):
        super().__init__()

        c = base_channels
        
        self.encoder = nn.Sequential(
            ConvBlock(3, c, activation="none", norm_type="none"),
            ConvBlock(c, c * 2, activation, norm_type),
            ConvBlock(c * 2, c * 4, activation, norm_type),
            ConvBlock(c * 4, c * 8, activation, norm_type),
        )

        self.feature_size = img_size // 16
        self.hidden_dim = c * 8 * self.feature_size * self.feature_size
        self.decoder_input_shape = (c * 8, self.feature_size, self.feature_size)

        self.flatten = nn.Flatten()
        self.fc_latent = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.hidden_dim)

        self.decoder = nn.Sequential(
            DeconvBlock(c * 8, c * 4, activation, norm_type),
            DeconvBlock(c * 4, c * 2, activation, norm_type),
            DeconvBlock(c * 2, c, activation, norm_type),
            nn.ConvTranspose2d(c, 3, kernel_size=4, stride=2, padding=1),
            get_output_activation(output_activation),
        )

    def encode(self, x):
        h = self.encoder(x)
        z = self.fc_latent(self.flatten(h))
        return z

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(z.size(0), *self.decoder_input_shape)
        return self.decoder(h)

    def forward(self, x, return_latent=False):
        z = self.encode(x)
        recon = self.decode(z)
        if return_latent:
            return recon, z
        return recon    
