import torch
import torch.nn as nn
from pokemon_diffusion.models.vae import ConvBlock, DeconvBlock
from pokemon_diffusion.models.helper import get_activation, get_output_activation, get_norm_layer

class ConditionEmbedding(nn.Module):
    def __init__(self, num_types, num_colors, num_shapes, type_emb_dim=32, color_emb_dim=16, shape_emb_dim=16):
        super().__init__()
        self.type_emb = nn.Embedding(num_types, type_emb_dim)
        self.color_emb = nn.Embedding(num_colors, color_emb_dim)
        self.shape_emb = nn.Embedding(num_shapes, shape_emb_dim)
        self.cond_dim = type_emb_dim + color_emb_dim + shape_emb_dim

    def forward(self, type_id, color_id, shape_id):
        type_e = self.type_emb(type_id)
        color_e = self.color_emb(color_id)
        shape_e = self.shape_emb(shape_id)
        return torch.cat([type_e, color_e, shape_e], dim=1)

class ConditionalEncoder(nn.Module):
    def __init__(self, img_size=96, latent_dim=512, num_types=20, num_colors=20, num_shapes=20,
                 type_emb_dim=32, color_emb_dim=16, shape_emb_dim=16,
                 activation="gelu", norm_type="group", base_channels=64):
        super().__init__()
        c = base_channels
        self.condition = ConditionEmbedding(num_types, num_colors, num_shapes, type_emb_dim, color_emb_dim, shape_emb_dim)
        self.encoder = nn.Sequential(
            ConvBlock(3, c, activation="none", norm_type="none"),
            ConvBlock(c, c * 2, activation, norm_type),
            ConvBlock(c * 2, c * 4, activation, norm_type),
            ConvBlock(c * 4, c * 8, activation, norm_type),
        )
        self.feature_size = img_size // 16
        self.hidden_dim = c * 8 * self.feature_size * self.feature_size
        self.fc_mu = nn.Linear(self.hidden_dim + self.condition.cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim + self.condition.cond_dim, latent_dim)

    def forward(self, x, type_id, color_id, shape_id):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        cond = self.condition(type_id, color_id, shape_id)
        h = torch.cat([h, cond], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

class ConditionalDecoder(nn.Module):
    def __init__(self, img_size=96, latent_dim=512, num_types=20, num_colors=20, num_shapes=20,
                 type_emb_dim=32, color_emb_dim=16, shape_emb_dim=16,
                 activation="gelu", norm_type="group", output_activation="sigmoid", base_channels=64):
        super().__init__()
        c = base_channels
        self.condition = ConditionEmbedding(num_types, num_colors, num_shapes, type_emb_dim, color_emb_dim, shape_emb_dim)
        self.feature_size = img_size // 16
        self.hidden_dim = c * 8 * self.feature_size * self.feature_size
        self.decoder_input_shape = (c * 8, self.feature_size, self.feature_size)
        self.fc_decode = nn.Linear(latent_dim + self.condition.cond_dim, self.hidden_dim)
        self.decoder = nn.Sequential(
            DeconvBlock(c * 8, c * 4, activation, norm_type),
            DeconvBlock(c * 4, c * 2, activation, norm_type),
            DeconvBlock(c * 2, c, activation, norm_type),
            DeconvBlock(c, 3, activation="none", norm_type="none"),
        )
        self.output_activation = get_output_activation(output_activation)

    def forward(self, z, type_id, color_id, shape_id):
        cond = self.condition(type_id, color_id, shape_id)
        z = torch.cat([z, cond], dim=1)
        h = self.fc_decode(z)
        h = h.view(z.size(0), *self.decoder_input_shape)
        return self.output_activation(self.decoder(h))

class CVAE(nn.Module):
    def __init__(self, img_size=96, latent_dim=512, num_types=20, num_colors=20, num_shapes=20,
                 type_emb_dim=32, color_emb_dim=16, shape_emb_dim=16,
                 activation="gelu", norm_type="group", output_activation="sigmoid", base_channels=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConditionalEncoder(
            img_size, latent_dim, num_types, num_colors, num_shapes,
            type_emb_dim, color_emb_dim, shape_emb_dim,
            activation, norm_type, base_channels
        )
        self.decoder = ConditionalDecoder(
            img_size, latent_dim, num_types, num_colors, num_shapes,
            type_emb_dim, color_emb_dim, shape_emb_dim,
            activation, norm_type, output_activation, base_channels
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, type_id, color_id, shape_id):
        mu, logvar = self.encoder(x, type_id, color_id, shape_id)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, type_id, color_id, shape_id)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, type_id, color_id, shape_id, device):
        type_id = type_id.to(device)
        color_id = color_id.to(device)
        shape_id = shape_id.to(device)
        z = torch.randn(type_id.size(0), self.latent_dim, device=device)
        return self.decoder(z, type_id, color_id, shape_id)