import torch
import torch.nn as nn

class SDLatentEncoderDecoder(nn.Module):
    def __init__(self, sd_vae):
        super().__init__()
        self.vae = sd_vae
        self.vae.requires_grad_(False)
        self.vae.eval()

    @torch.no_grad()
    def encode(self, x):
        z = self.vae.encode(x).latent_dist.mean
        return z * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode(self, z):
        z = z.to(dtype=next(self.vae.parameters()).dtype)
        return self.vae.decode(z / self.vae.config.scaling_factor).sample


class ConditionalLatentPrior(nn.Module):
    def __init__(self, num_types, num_colors, num_shapes, latent_shape=(4, 64, 64)):
        super().__init__()
        self.latent_shape = latent_shape
        latent_dim = latent_shape[0] * latent_shape[1] * latent_shape[2]

        self.type_emb = nn.Embedding(num_types, 32)
        self.color_emb = nn.Embedding(num_colors, 16)
        self.shape_emb = nn.Embedding(num_shapes, 16)

        self.net = nn.Sequential(
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
        )

        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, type_id, color_id, shape_id):
        cond = torch.cat([
            self.type_emb(type_id),
            self.color_emb(color_id),
            self.shape_emb(shape_id),
        ], dim=1)

        h = self.net(cond)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-10, 10)
        return mu, logvar

    def sample(self, type_id, color_id, shape_id):
        mu, logvar = self(type_id, color_id, shape_id)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)
        return z.view(len(type_id), *self.latent_shape)