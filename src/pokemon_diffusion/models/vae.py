import torch
import torch.nn as nn
from pokemon_diffusion.models.helper import (
    get_activation,
    get_output_activation,
    get_norm_layer,
)


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


class Encoder(nn.Module):
    def __init__(self, img_size=96, latent_dim=128, activation="relu", norm_type="none", base_channels=32):
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
        self.fc_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, latent_dim)


    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    

class Decoder(nn.Module):
    def __init__(self, img_size=96, latent_dim=128, activation="relu", norm_type="none",
                 output_activation="tanh", base_channels=32):
        super().__init__()
        c = base_channels
        self.feature_size = img_size // 16
        self.hidden_dim = c * 8 * self.feature_size * self.feature_size
        self.decoder_input_shape = (c * 8, self.feature_size, self.feature_size)

        self.fc_decode = nn.Linear(latent_dim, self.hidden_dim)

        self.decoder = nn.Sequential(
            DeconvBlock(c * 8, c * 4, activation, norm_type),
            DeconvBlock(c * 4, c * 2, activation, norm_type),
            DeconvBlock(c * 2, c, activation, norm_type),
            DeconvBlock(c, 3, activation="none", norm_type="none"),
        )

        self.output_activation = get_output_activation(output_activation)

    def forward(self, z):
        h = self.fc_decode(z)
        h = h.view(z.size(0), *self.decoder_input_shape)
        x_recon = self.decoder(h)
        x_recon = self.output_activation(x_recon)
        return x_recon


class VAE(nn.Module):
    def __init__(self, img_size=96, latent_dim=128, activation="relu", norm_type="none",
                 output_activation="tanh", base_channels=32):
        super().__init__()
        self.encoder = Encoder(img_size, latent_dim, activation, norm_type, base_channels)
        self.decoder = Decoder(img_size, latent_dim, activation, norm_type, output_activation, base_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar