import os

import torch
import torch.nn as nn
from tqdm import tqdm
from .loss import get_loss_fn
from torchvision.utils import make_grid
import wandb


def to_vis_range(x, params):
    norm_range = params["data"]["normalization_range"]

    if norm_range == "minus_one_to_one":
        img = (x + 1) / 2
    elif norm_range == "zero_to_one":
        img = x
    else:
        raise ValueError(f"Unknown normalization_range: {norm_range}")

    return img.clamp(0, 1)
    

def log_reconstruction_images(model, val_loader, device, epoch, params, num_images=8):
    model.eval()

    batch = next(iter(val_loader))
    x = batch["image"].to(device) if isinstance(batch, dict) else batch[0].to(device)
    x = x[:num_images]

    with torch.no_grad():
        recon = model(x)

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

    return grid


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        x = batch["image"].to(device) if isinstance(batch, dict) else batch[0].to(device) # Support both dict and tuple batch formats

        optimizer.zero_grad()
        recon = model(x)
        loss = loss_fn(recon, x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        x = batch["image"].to(device) if isinstance(batch, dict) else batch[0].to(device)
        recon = model(x)
        loss = loss_fn(recon, x)
        total_loss += loss.item() * x.size(0)

    return total_loss / len(dataloader.dataset)


def train_ae_model(model, train_loader, val_loader, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    experiment_name = params.get("experiment_name", "ae_experiment")

    wandb.init(project="pokemon_autoencoder", name=experiment_name, config=params,)

    loss_fn = get_loss_fn(params["training"]["loss"], params)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["training"]["learning_rate"],
        weight_decay=params["training"].get("weight_decay", 0.0),
    )

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(params["training"]["epochs"]):

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(
            f"Epoch {epoch + 1}/{params['training']['epochs']} | "
            f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}"
        )

    result_img =log_reconstruction_images(model, val_loader, device, epoch + 1, params)

    save_dir = "/home/young/projects/pokemon_diffusion/data/06_models/AE"
    model_path = os.path.join(save_dir, f"{experiment_name}.pt")
    
    torch.save(model.state_dict(), model_path)

    artifact = wandb.Artifact(name=f"{experiment_name}_model", type="model",)
    artifact.add_file(model_path)

    wandb.log_artifact(artifact)    
    wandb.finish()

    return history, result_img