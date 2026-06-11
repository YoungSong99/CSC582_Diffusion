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

def vqvae_loss(recon, x, vq_loss, recon_loss_fn, vq_weight=1.0):
    recon_loss = recon_loss_fn(recon, x)
    total = recon_loss + vq_weight * vq_loss
    return total, recon_loss, vq_loss

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

def train_one_epoch(model, dataloader, optimizer, recon_loss_fn, device, vq_weight):
    model.train()
    total_loss, total_recon, total_vq, total_used_codes = 0.0, 0.0, 0.0, 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        x = batch["image"].to(device) if isinstance(batch, dict) else batch[0].to(device)

        optimizer.zero_grad()
        recon, vq_loss, indices = model(x)
        loss, recon_loss, vq = vqvae_loss(recon, x, vq_loss, recon_loss_fn, vq_weight)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_recon += recon_loss.item() * bs
        total_vq += vq.item() * bs
        total_used_codes += indices.view(-1).unique().numel()

    n = len(dataloader.dataset)
    avg_used_codes = total_used_codes / len(dataloader)
    return total_loss / n, total_recon / n, total_vq / n, avg_used_codes

@torch.no_grad()
def evaluate(model, dataloader, recon_loss_fn, device, vq_weight):
    model.eval()
    total_loss, total_recon, total_vq, total_used_codes = 0.0, 0.0, 0.0, 0.0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        x = batch["image"].to(device) if isinstance(batch, dict) else batch[0].to(device)

        recon, vq_loss, indices = model(x)
        loss, recon_loss, vq = vqvae_loss(recon, x, vq_loss, recon_loss_fn, vq_weight)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_recon += recon_loss.item() * bs
        total_vq += vq.item() * bs
        total_used_codes += indices.view(-1).unique().numel()

    n = len(dataloader.dataset)
    avg_used_codes = total_used_codes / len(dataloader)
    return total_loss / n, total_recon / n, total_vq / n, avg_used_codes

def train_vqvae_model(model, train_loader, val_loader, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    experiment_name = params.get("experiment_name", "vqvae_experiment")
    wandb.init(project="pokemon_vqvae", name=experiment_name, config=params,)

    recon_loss_fn = get_loss_fn(params["training"]["loss"], params).to(device)
    vq_weight = params["training"].get("vq_weight", 1.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["training"]["learning_rate"],
        weight_decay=params["training"].get("weight_decay", 0.0),
    )

    history = {
        "train_loss": [], "train_recon_loss": [], "train_vq_loss": [], "train_used_codes": [],
        "val_loss": [], "val_recon_loss": [], "val_vq_loss": [], "val_used_codes": [],
    }

    for epoch in range(params["training"]["epochs"]):
        train_loss, train_recon, train_vq, train_used_codes = train_one_epoch(
            model, train_loader, optimizer, recon_loss_fn, device, vq_weight
        )
        val_loss, val_recon, val_vq, val_used_codes = evaluate(
            model, val_loader, recon_loss_fn, device, vq_weight
        )

        history["train_loss"].append(train_loss)
        history["train_recon_loss"].append(train_recon)
        history["train_vq_loss"].append(train_vq)
        history["train_used_codes"].append(train_used_codes)
        history["val_loss"].append(val_loss)
        history["val_recon_loss"].append(val_recon)
        history["val_vq_loss"].append(val_vq)
        history["val_used_codes"].append(val_used_codes)

        wandb.log({
            "train_loss": float(train_loss),
            "train_recon_loss": float(train_recon),
            "train_vq_loss": float(train_vq),
            "train_used_codes": float(train_used_codes),
            "val_loss": float(val_loss),
            "val_recon_loss": float(val_recon),
            "val_vq_loss": float(val_vq),
            "val_used_codes": float(val_used_codes),
            "vq_weight": float(vq_weight),
            "epoch": epoch + 1,
        })

        print(
            f"Epoch {epoch + 1}/{params['training']['epochs']} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_recon: {train_recon:.4f} | "
            f"train_vq: {train_vq:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_recon: {val_recon:.4f} | "
            f"val_vq: {val_vq:.4f}"
        )

    result_img = log_reconstruction_images(model, val_loader, device, epoch + 1, params)

    save_dir = "/home/young/projects/pokemon_diffusion/data/06_models/VQVAE"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"{experiment_name}.pt")
    torch.save(model.state_dict(), model_path)

    artifact = wandb.Artifact(name=f"{experiment_name}_model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    wandb.finish()

    return history, result_img