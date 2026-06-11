import os
import torch
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import wandb
from .loss import get_loss_fn

def to_vis_range(x, params):
    norm_range = params["data"]["normalization_range"]
    if norm_range == "minus_one_to_one":
        img = (x + 1) / 2
    elif norm_range == "zero_to_one":
        img = x
    else:
        raise ValueError(f"Unknown normalization_range: {norm_range}")
    return img.clamp(0, 1)

def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def vae_loss(recon, x, mu, logvar, recon_loss_fn, beta=1.0):
    recon_loss = recon_loss_fn(recon, x)
    kl = kl_loss(mu, logvar)
    total = recon_loss + beta * kl
    return total, recon_loss, kl

def log_reconstruction_images(model, val_loader, device, epoch, params, num_images=8):
    model.eval()
    batch = next(iter(val_loader))
    x = batch["image"].to(device) if isinstance(batch, dict) else batch[0].to(device)
    x = x[:num_images]

    with torch.no_grad():
        recon, _, _ = model(x)

    x_vis = to_vis_range(x, params)
    recon_vis = to_vis_range(recon, params)
    comparison = torch.cat([x_vis, recon_vis], dim=0)
    grid = make_grid(comparison, nrow=num_images)

    wandb.log({
        "reconstructions": wandb.Image(
            grid,
            caption=f"Top: original | Bottom: reconstruction | Epoch {epoch}"
        )
    }, step=epoch)

    return to_pil_image(grid.cpu())

def train_one_epoch(model, dataloader, optimizer, recon_loss_fn, device, beta):
    model.train()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        x = batch["image"].to(device) if isinstance(batch, dict) else batch[0].to(device)

        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss, recon_loss, kl = vae_loss(recon, x, mu, logvar, recon_loss_fn, beta)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_recon += recon_loss.item() * bs
        total_kl += kl.item() * bs

    n = len(dataloader.dataset)
    return total_loss / n, total_recon / n, total_kl / n

@torch.no_grad()
def evaluate(model, dataloader, recon_loss_fn, device, beta):
    model.eval()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        x = batch["image"].to(device) if isinstance(batch, dict) else batch[0].to(device)
        recon, mu, logvar = model(x)
        loss, recon_loss, kl = vae_loss(recon, x, mu, logvar, recon_loss_fn, beta)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_recon += recon_loss.item() * bs
        total_kl += kl.item() * bs

    n = len(dataloader.dataset)
    return total_loss / n, total_recon / n, total_kl / n

def train_vae_model(model, train_loader, val_loader, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    experiment_name = params.get("experiment_name", "vae_experiment")
    wandb.init(project="pokemon_vae", name=experiment_name, config=params,)

    recon_loss_fn = get_loss_fn(params["training"]["loss"], params)
    beta = params["training"].get("beta", 1.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["training"]["learning_rate"],
        weight_decay=params["training"].get("weight_decay", 0.0),
    )

    history = {
        "train_loss": [], "train_recon_loss": [], "train_kl_loss": [],
        "val_loss": [], "val_recon_loss": [], "val_kl_loss": [],
    }

    for epoch in range(params["training"]["epochs"]):
        train_loss, train_recon, train_kl = train_one_epoch(
            model, train_loader, optimizer, recon_loss_fn, device, beta
        )
        val_loss, val_recon, val_kl = evaluate(
            model, val_loader, recon_loss_fn, device, beta
        )

        history["train_loss"].append(train_loss)
        history["train_recon_loss"].append(train_recon)
        history["train_kl_loss"].append(train_kl)
        history["val_loss"].append(val_loss)
        history["val_recon_loss"].append(val_recon)
        history["val_kl_loss"].append(val_kl)

        wandb.log({
            "train_loss": float(train_loss),
            "train_recon_loss": float(train_recon),
            "train_kl_loss": float(train_kl),
            "val_loss": float(val_loss),
            "val_recon_loss": float(val_recon),
            "val_kl_loss": float(val_kl),
            "beta": float(beta),
            "epoch": epoch + 1,
        })

        print(
            f"Epoch {epoch + 1}/{params['training']['epochs']} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_recon: {train_recon:.4f} | "
            f"train_kl: {train_kl:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_recon: {val_recon:.4f} | "
            f"val_kl: {val_kl:.4f}"
        )

    result_img = log_reconstruction_images(model, val_loader, device, epoch + 1, params)

    save_dir = "/home/young/projects/pokemon_diffusion/data/06_models/VAE"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"{experiment_name}.pt")
    torch.save(model.state_dict(), model_path)

    artifact = wandb.Artifact(name=f"{experiment_name}_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    wandb.finish()

    return history, result_img