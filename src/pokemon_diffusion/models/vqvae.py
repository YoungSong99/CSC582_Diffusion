import torch
import torch.nn as nn
import torch.nn.functional as F
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


class ResBlock(nn.Module):
    def __init__(self, channels, activation="gelu", norm_type="group"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            get_norm_layer(norm_type, channels),
            get_activation(activation),
            nn.Conv2d(channels, channels, 3, 1, 1),
            get_norm_layer(norm_type, channels),
        )
        self.act = get_activation(activation)

    def forward(self, x):
        return self.act(x + self.block(x))


class UpsampleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation="gelu", norm_type="group"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            get_norm_layer(norm_type, out_ch),
            get_activation(activation),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, img_size=96, activation="relu", norm_type="none", base_channels=32):
        super().__init__()
        c = base_channels
        self.out_channels = c * 4
        self.encoder = nn.Sequential(
            ConvBlock(3, c, activation="none", norm_type="none"),
            ConvBlock(c, c * 2, activation, norm_type),
            ConvBlock(c * 2, c * 4, activation, norm_type),
        )

    def forward(self, x):
        return self.encoder(x)
    

class Decoder(nn.Module):
    def __init__(self, activation="relu", norm_type="none", output_activation="tanh", base_channels=32):
        super().__init__()
        c = base_channels

        self.decoder = nn.Sequential(
            ResBlock(c * 4, activation, norm_type),
            UpsampleConvBlock(c * 4, c * 2, activation, norm_type),
            ResBlock(c * 2, activation, norm_type),
            UpsampleConvBlock(c * 2, c, activation, norm_type),
            ResBlock(c, activation, norm_type),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c, 3, 3, 1, 1),
        )

        self.output_activation = get_output_activation(output_activation)

    def forward(self, h):
        return self.output_activation(self.decoder(h))
    

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, z):
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_perm.view(-1, self.embedding_dim)

        distances = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )

        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices).view(z_perm.shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        quantized = z + (quantized - z).detach()
        indices = indices.view(z.size(0), z.size(2), z.size(3))
        used_codes = torch.unique(indices).numel()

        return quantized, vq_loss, indices

class VQVAE(nn.Module):
    def __init__(
        self, img_size=96, latent_channels=256, num_embeddings=512,
        activation="relu", norm_type="none", output_activation="tanh",
        base_channels=32, commitment_cost=0.25
    ):
        super().__init__()
        c = base_channels
        self.encoder = Encoder(img_size, activation, norm_type, base_channels)
        self.pre_vq = nn.Conv2d(c * 4, latent_channels, kernel_size=1)
        self.vq = VectorQuantizer(num_embeddings, latent_channels, commitment_cost)
        self.post_vq = nn.Conv2d(latent_channels, c * 4, kernel_size=1)
        self.decoder = Decoder(activation, norm_type, output_activation, base_channels)

    def encode(self, x):
        return self.pre_vq(self.encoder(x))

    def decode_from_quantized(self, quantized):
        return self.decoder(self.post_vq(quantized))

    def forward(self, x):
        z = self.encode(x)
        quantized, vq_loss, indices = self.vq(z)
        x_recon = self.decode_from_quantized(quantized)
        return x_recon, vq_loss, indices