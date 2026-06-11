import torch
import torch.nn as nn

class ConditionEmbedding(nn.Module):
    def __init__(self, num_types, num_colors, num_shapes, type_emb_dim=32, color_emb_dim=16, shape_emb_dim=16):
        super().__init__()
        self.type_emb = nn.Embedding(num_types, type_emb_dim)
        self.color_emb = nn.Embedding(num_colors, color_emb_dim)
        self.shape_emb = nn.Embedding(num_shapes, shape_emb_dim)
        self.cond_dim = type_emb_dim + color_emb_dim + shape_emb_dim

    def forward(self, type_id, color_id, shape_id):
        return torch.cat([
            self.type_emb(type_id),
            self.color_emb(color_id),
            self.shape_emb(shape_id),
        ], dim=1)

class ConditionProjection(nn.Module):
    def __init__(self, cond_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, cond):
        return self.net(cond)
    

class ConditionalPretrainedVAE(nn.Module):
    def __init__(self, vae, latent_dim=128, num_types=20, num_colors=20, num_shapes=20, cond_scale=0.1):
        super().__init__()
        self.vae = vae
        self.cond_scale = cond_scale

        for p in self.vae.parameters():
            p.requires_grad = False

        self.condition = ConditionEmbedding(num_types, num_colors, num_shapes)
        self.cond_proj = ConditionProjection(self.condition.cond_dim, latent_dim)

    def forward(self, x, type_id, color_id, shape_id):
        with torch.no_grad():
            mu, logvar = self.vae.encoder(x)
            z = self.vae.reparameterize(mu, logvar)

        cond = self.condition(type_id, color_id, shape_id)
        cond_z = self.cond_proj(cond)

        z = z + self.cond_scale * cond_z
        recon = self.vae.decoder(z)
        return recon

    @torch.no_grad()
    def sample(self, type_id, color_id, shape_id, device):
        type_id = type_id.to(device)
        color_id = color_id.to(device)
        shape_id = shape_id.to(device)

        z = torch.randn(type_id.size(0), self.cond_proj.net[-1].out_features, device=device)
        cond = self.condition(type_id, color_id, shape_id)
        z = z + self.cond_scale * self.cond_proj(cond)

        return self.vae.decoder(z)
    

class ConditionalPretrainedVQVAE(nn.Module):
    def __init__(self, vqvae, latent_channels=128, num_types=20, num_colors=20, num_shapes=20, cond_scale=0.1):
        super().__init__()
        self.vqvae = vqvae
        self.cond_scale = cond_scale

        for p in self.vqvae.parameters():
            p.requires_grad = False

        self.condition = ConditionEmbedding(num_types, num_colors, num_shapes)
        self.cond_proj = ConditionProjection(self.condition.cond_dim, latent_channels)

    def forward(self, x, type_id, color_id, shape_id):
        with torch.no_grad():
            z = self.vqvae.encode(x)
            quantized, _, indices = self.vqvae.vq(z)

        cond = self.condition(type_id, color_id, shape_id)
        cond = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)

        quantized = quantized + self.cond_scale * cond
        recon = self.vqvae.decode_from_quantized(quantized)
        return recon, indices

    @torch.no_grad()
    def sample(self, type_id, color_id, shape_id, device, h=12, w=12):
        type_id = type_id.to(device)
        color_id = color_id.to(device)
        shape_id = shape_id.to(device)

        indices = torch.randint(
            0, self.vqvae.vq.num_embeddings,
            (type_id.size(0), h, w),
            device=device,
        )

        quantized = self.vqvae.vq.embedding(indices)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        cond = self.condition(type_id, color_id, shape_id)
        cond = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)

        quantized = quantized + self.cond_scale * cond
        return self.vqvae.decode_from_quantized(quantized)
    

class ConditionalSDAutoencoder(nn.Module):
    def __init__(self, sd_vae, num_types=20, num_colors=20, num_shapes=20, cond_scale=0.1):
        super().__init__()
        self.vae = sd_vae
        self.cond_scale = cond_scale

        for p in self.vae.parameters():
            p.requires_grad = False

        self.condition = ConditionEmbedding(num_types, num_colors, num_shapes)
        self.cond_proj = ConditionProjection(self.condition.cond_dim, 4)

    def forward(self, x, type_id, color_id, shape_id):
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor

        cond = self.condition(type_id, color_id, shape_id)
        cond = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)

        latent = latent + self.cond_scale * cond
        recon = self.vae.decode(latent / self.vae.config.scaling_factor).sample

        return recon

    @torch.no_grad()
    def sample(self, type_id, color_id, shape_id, device, h=64, w=64):
        type_id = type_id.to(device)
        color_id = color_id.to(device)
        shape_id = shape_id.to(device)

        latent = torch.randn(type_id.size(0), 4, h, w, device=device)
        cond = self.condition(type_id, color_id, shape_id)
        cond = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)

        latent = latent + self.cond_scale * cond
        return self.vae.decode(latent / self.vae.config.scaling_factor).sample