import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from diffusers import AutoencoderKL
from .loss import get_loss_fn

class ConditionalSDAutoencoder(nn.Module):
    def __init__(self, sd_vae, num_types, num_colors, num_shapes, cond_scale=0.05):
        super().__init__()
        self.vae = sd_vae
        self.cond_scale = cond_scale

        for p in self.vae.parameters():
            p.requires_grad = False

        self.type_emb = nn.Embedding(num_types, 32)
        self.color_emb = nn.Embedding(num_colors, 16)
        self.shape_emb = nn.Embedding(num_shapes, 16)

        self.cond_proj = nn.Sequential(
            nn.Linear(64, 4),
            nn.GELU(),
            nn.Linear(4, 4),
        )

    def forward(self, x, type_id, color_id, shape_id):
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor

        cond = torch.cat([
            self.type_emb(type_id),
            self.color_emb(color_id),
            self.shape_emb(shape_id),
        ], dim=1)

        cond = self.cond_proj(cond)
        cond = cond.to(device=latent.device, dtype=latent.dtype)
        cond = cond.unsqueeze(-1).unsqueeze(-1)

        latent = latent + self.cond_scale * cond
        latent = latent.to(dtype=next(self.vae.parameters()).dtype)

        recon = self.vae.decode(latent / self.vae.config.scaling_factor).sample
        return recon
    
    @torch.no_grad()
    def edit_color_from_image(self, x, type_id, shape_id, color_ids):
        latent = self.vae.encode(x).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        latent = latent.repeat(len(color_ids), 1, 1, 1)

        type_id = torch.full((len(color_ids),), int(type_id), device=x.device).long()
        shape_id = torch.full((len(color_ids),), int(shape_id), device=x.device).long()
        color_id = torch.tensor(color_ids, device=x.device).long()

        cond = torch.cat([
            self.type_emb(type_id),
            self.color_emb(color_id),
            self.shape_emb(shape_id),
        ], dim=1)

        cond = self.cond_proj(cond).to(device=latent.device, dtype=latent.dtype)
        cond = cond.unsqueeze(-1).unsqueeze(-1)

        latent = latent + self.cond_scale * cond
        recon = self.vae.decode(latent / self.vae.config.scaling_factor).sample
        return recon
    

@torch.no_grad()
def log_same_image_different_colors(model, val_loader, device, dtype, epoch, params, num_colors=None):
    model.eval()
    batch = next(iter(val_loader))

    x = batch["image"][:1].to(device, dtype=dtype)
    type_id = batch["type_id"][0].item()
    shape_id = batch["shape_id"][0].item()

    x = to_sd_range(x, params)
    x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)

    if num_colors is None:
        num_colors = params["model"]["num_colors"]

    color_ids = list(range(num_colors))

    recon = model.edit_color_from_image(
        x=x,
        type_id=type_id,
        shape_id=shape_id,
        color_ids=color_ids,
    )

    x_vis = to_vis_range(x)
    recon_vis = to_vis_range(recon)

    grid = make_grid(torch.cat([x_vis, recon_vis], dim=0), nrow=8)

    wandb.log({
        "same_image_different_colors": wandb.Image(
            grid,
            caption=f"Original + same latent/type/shape, different colors | Epoch {epoch}"
        )
    }, step=epoch)

    return to_pil_image(grid.cpu())

def to_sd_range(x, params):
    norm_range = params["data"]["normalization_range"]
    if norm_range == "zero_to_one":
        return x * 2 - 1
    if norm_range == "minus_one_to_one":
        return x
    raise ValueError(f"Unknown normalization_range: {norm_range}")

def to_vis_range(x):
    return ((x + 1) / 2).clamp(0, 1)

def get_batch(batch, device, dtype, params):
    x = batch["image"].to(device, dtype=dtype)
    type_id = batch["type_id"].to(device).long()
    color_id = batch["color_id"].to(device).long()
    shape_id = batch["shape_id"].to(device).long()

    x = to_sd_range(x, params)
    x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
    return x, type_id, color_id, shape_id

def train_one_epoch(model, dataloader, optimizer, recon_loss_fn, device, dtype, params):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        x, type_id, color_id, shape_id = get_batch(batch, device, dtype, params)

        optimizer.zero_grad()
        recon = model(x, type_id, color_id, shape_id)
        loss = recon_loss_fn(recon, x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def evaluate(model, dataloader, recon_loss_fn, device, dtype, params):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        x, type_id, color_id, shape_id = get_batch(batch, device, dtype, params)
        recon = model(x, type_id, color_id, shape_id)
        loss = recon_loss_fn(recon, x)
        total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def log_reconstruction_images(model, val_loader, device, dtype, epoch, params, num_images=8):
    model.eval()
    batch = next(iter(val_loader))
    x, type_id, color_id, shape_id = get_batch(batch, device, dtype, params)

    x = x[:num_images]
    type_id = type_id[:num_images]
    color_id = color_id[:num_images]
    shape_id = shape_id[:num_images]

    recon = model(x, type_id, color_id, shape_id)

    x_vis = to_vis_range(x)
    recon_vis = to_vis_range(recon)
    grid = make_grid(torch.cat([x_vis, recon_vis], dim=0), nrow=num_images)

    wandb.log({
        "conditional_sd_reconstructions": wandb.Image(
            grid,
            caption=f"Top: original | Bottom: conditional SD-VAE recon | Epoch {epoch}"
        )
    }, step=epoch)

    return to_pil_image(grid.cpu())

def train_conditional_sdvae_model(train_loader, val_loader, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    experiment_name = params.get("experiment_name", "conditional_sdvae")

    wandb.init(project="pokemon_cvae", name=experiment_name, config=params)

    sd_vae = AutoencoderKL.from_pretrained(
        params["pretrained"].get("name", "stabilityai/sd-vae-ft-mse"),
        torch_dtype=dtype,
    ).to(device)

    sd_vae.eval()

    model = ConditionalSDAutoencoder(
        sd_vae=sd_vae,
        num_types=params["model"]["num_types"],
        num_colors=params["model"]["num_colors"],
        num_shapes=params["model"]["num_shapes"],
        cond_scale=params["model"].get("cond_scale", 0.05),
    ).to(device)

    recon_loss_fn = get_loss_fn(params["training"]["loss"], params).to(device)

    optimizer = torch.optim.AdamW(
        list(model.type_emb.parameters()) +
        list(model.color_emb.parameters()) +
        list(model.shape_emb.parameters()) +
        list(model.cond_proj.parameters()),
        lr=params["training"]["learning_rate"],
        weight_decay=params["training"].get("weight_decay", 0.0),
    )

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(params["training"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, recon_loss_fn, device, dtype, params)
        val_loss = evaluate(model, val_loader, recon_loss_fn, device, dtype, params)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        wandb.log({
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "cond_scale": float(model.cond_scale),
            "epoch": epoch + 1,
        })

        print(
            f"Epoch {epoch + 1}/{params['training']['epochs']} | "
            f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}"
        )

    result_img = log_same_image_different_colors(model, val_loader, device, dtype, epoch + 1, params)

    save_dir = "/home/young/projects/pokemon_diffusion/data/06_models/ConditionalSDVAE"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"{experiment_name}.pt")
    torch.save(model.state_dict(), model_path)

    artifact = wandb.Artifact(name=f"{experiment_name}_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    wandb.finish()

    return model, history, result_img