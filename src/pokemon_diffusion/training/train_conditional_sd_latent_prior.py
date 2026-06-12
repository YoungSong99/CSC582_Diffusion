import os, torch, wandb
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from diffusers import AutoencoderKL
from pokemon_diffusion.models.conditional_sd_latent_prior import (
    SDLatentEncoderDecoder,
    ConditionalLatentPrior,
)

def to_sd_range(x, params):
    r = params["data"]["normalization_range"]
    if r == "zero_to_one": return x * 2 - 1
    if r == "minus_one_to_one": return x
    raise ValueError(f"Unknown normalization_range: {r}")

def to_vis_range(x):
    return ((x + 1) / 2).clamp(0, 1)

def get_batch(batch, device, dtype, params):
    x = batch["image"].to(device, dtype=dtype)
    x = to_sd_range(x, params)
    x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
    return (
        x,
        batch["type_id"].to(device).long(),
        batch["color_id"].to(device).long(),
        batch["shape_id"].to(device).long(),
    )


@torch.no_grad()
def precompute_latents(dataloader, sd_latent, device, dtype, params, save_path):
    sd_latent.eval()
    latents, type_ids, color_ids, shape_ids = [], [], [], []

    for batch in tqdm(dataloader, desc="Precomputing latents"):
        x, t, c, s = get_batch(batch, device, dtype, params)
        z = sd_latent.encode(x)
        latents.append(z.cpu())
        type_ids.append(t.cpu())
        color_ids.append(c.cpu())
        shape_ids.append(s.cpu())

    cache = {
        "latent": torch.cat(latents),
        "type_id": torch.cat(type_ids),
        "color_id": torch.cat(color_ids),
        "shape_id": torch.cat(shape_ids),
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(cache, save_path)
    print(f"Saved latent cache to {save_path}")


class LatentCacheDataset(Dataset):
    def __init__(self, cache_path):
        cache = torch.load(cache_path, map_location="cpu")
        self.latent = cache["latent"]
        self.type_id = cache["type_id"]
        self.color_id = cache["color_id"]
        self.shape_id = cache["shape_id"]

    def __len__(self):
        return len(self.latent)

    def __getitem__(self, idx):
        return {
            "latent": self.latent[idx],
            "type_id": self.type_id[idx],
            "color_id": self.color_id[idx],
            "shape_id": self.shape_id[idx],
        }
    

def train_one_epoch_prior(model, loader, optimizer, device, dtype, beta=1e-4):
    model.train()
    total = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        z = batch["latent"].to(device, dtype=dtype)
        t = batch["type_id"].to(device).long()
        c = batch["color_id"].to(device).long()
        s = batch["shape_id"].to(device).long()

        target = z.flatten(1)
        mu, logvar = model(t, c, s)

        recon_loss = F.mse_loss(mu, target)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item() * z.size(0)

    return total / len(loader.dataset)


@torch.no_grad()
def evaluate_prior(model, loader, device, dtype, beta=1e-4):
    model.eval()
    total = 0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        z = batch["latent"].to(device, dtype=dtype)
        t = batch["type_id"].to(device).long()
        c = batch["color_id"].to(device).long()
        s = batch["shape_id"].to(device).long()

        target = z.flatten(1)
        mu, logvar = model(t, c, s)

        recon_loss = F.mse_loss(mu, target)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss

        total += loss.item() * z.size(0)

    return total / len(loader.dataset)


@torch.no_grad()
def log_generated_samples(prior, sd_latent, epoch, device, dtype, params, num_images=8):
    prior.eval()

    t = torch.randint(0, params["model"]["num_types"], (num_images,), device=device)
    c = torch.randint(0, params["model"]["num_colors"], (num_images,), device=device)
    s = torch.randint(0, params["model"]["num_shapes"], (num_images,), device=device)

    z = prior.sample(t, c, s).to(device, dtype=dtype)
    img = sd_latent.decode(z)

    grid = make_grid(to_vis_range(img), nrow=num_images)

    wandb.log({
        "generated_from_labels": wandb.Image(
            grid,
            caption=f"Generated from type/color/shape labels | Epoch {epoch}"
        )
    }, step=epoch)

    return to_pil_image(grid.cpu())


@torch.no_grad()
def compute_color_centroids(loader, num_colors):
    sums, counts = {}, {}

    for batch in tqdm(loader, desc="Computing color centroids"):
        z = batch["latent"]
        c = batch["color_id"]

        for i in range(len(z)):
            color = int(c[i])
            sums[color] = sums.get(color, 0) + z[i]
            counts[color] = counts.get(color, 0) + 1

    return {
        color: sums[color] / counts[color]
        for color in range(num_colors)
        if color in sums
    }


@torch.no_grad()
def log_same_latent_different_colors(sd_latent, loader, centroids, epoch, device, dtype, params, strength=1.0):
    batch = next(iter(loader))

    z = batch["latent"][:1].to(device, dtype=dtype)
    source_color = int(batch["color_id"][0])
    num_colors = params["model"]["num_colors"]

    imgs = [sd_latent.decode(z)]

    for target_color in range(num_colors):
        if target_color not in centroids or source_color not in centroids:
            continue

        direction = centroids[target_color] - centroids[source_color]
        direction = direction.unsqueeze(0).to(device, dtype=dtype)
        z_edit = z + strength * direction
        imgs.append(sd_latent.decode(z_edit))

    grid = make_grid(to_vis_range(torch.cat(imgs)), nrow=8)

    wandb.log({
        "same_latent_different_colors": wandb.Image(
            grid,
            caption=f"Original + centroid color edits | Epoch {epoch}"
        )
    }, step=epoch)

    return to_pil_image(grid.cpu())


def train_conditional_sd_latent_prior(train_loader, val_loader, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    experiment_name = params.get("experiment_name", "conditional_sd_latent_prior")

    wandb.init(project="pokemon_cvae", name=experiment_name, config=params)

    sd_vae = AutoencoderKL.from_pretrained(
        params["pretrained"].get("name", "stabilityai/sd-vae-ft-mse"),
        torch_dtype=dtype,
    ).to(device)

    sd_latent = SDLatentEncoderDecoder(sd_vae).to(device)

    cache_dir = "/home/young/projects/pokemon_diffusion/data/05_model_input/latent_cache"
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, f"{experiment_name}_train.pt")
    val_cache = os.path.join(cache_dir, f"{experiment_name}_val.pt")

    if not os.path.exists(train_cache):
        precompute_latents(train_loader, sd_latent, device, dtype, params, train_cache)
    if not os.path.exists(val_cache):
        precompute_latents(val_loader, sd_latent, device, dtype, params, val_cache)

    train_ds = LatentCacheDataset(train_cache)
    val_ds = LatentCacheDataset(val_cache)

    train_latent_loader = DataLoader(
        train_ds,
        batch_size=params["data"]["batch_size"],
        shuffle=True,
        num_workers=params["data"].get("num_workers", 2),
    )

    val_latent_loader = DataLoader(
        val_ds,
        batch_size=params["data"]["batch_size"],
        shuffle=False,
        num_workers=params["data"].get("num_workers", 2),
    )

    prior = ConditionalLatentPrior(
        num_types=params["model"]["num_types"],
        num_colors=params["model"]["num_colors"],
        num_shapes=params["model"]["num_shapes"],
        latent_shape=(4, 64, 64),
    ).to(device)

    optimizer = torch.optim.AdamW(
        prior.parameters(),
        lr=params["training"]["learning_rate"],
        weight_decay=params["training"].get("weight_decay", 0.0),
    )

    beta = params["training"].get("kl_beta", 1e-4)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, params["training"]["epochs"] + 1):
        train_loss = train_one_epoch_prior(prior, train_latent_loader, optimizer, device, dtype, beta)
        val_loss = evaluate_prior(prior, val_latent_loader, device, dtype, beta)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        wandb.log({
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "kl_beta": float(beta),
            "epoch": epoch,
        }, step=epoch)

        print(f"Epoch {epoch}/{params['training']['epochs']} | train_loss: {train_loss:.5f} | val_loss: {val_loss:.5f}")

        if epoch % params["training"].get("log_every", 10) == 0:
            log_generated_samples(prior, sd_latent, epoch, device, dtype, params)

    centroids = compute_color_centroids(train_latent_loader, params["model"]["num_colors"])
    result_img = log_same_latent_different_colors(
        sd_latent, val_latent_loader, centroids, epoch, device, dtype, params,
        strength=params["model"].get("color_edit_strength", 1.0),
    )

    save_dir = "/home/young/projects/pokemon_diffusion/data/06_models/ConditionalSDLatentPrior"
    os.makedirs(save_dir, exist_ok=True)

    prior_path = os.path.join(save_dir, f"{experiment_name}_prior.pt")
    centroid_path = os.path.join(save_dir, f"{experiment_name}_color_centroids.pt")

    torch.save(prior.state_dict(), prior_path)
    torch.save(centroids, centroid_path)

    artifact = wandb.Artifact(name=f"{experiment_name}_model", type="model")
    artifact.add_file(prior_path)
    artifact.add_file(centroid_path)
    wandb.log_artifact(artifact)
    wandb.finish()

    return prior, sd_latent, history, result_img