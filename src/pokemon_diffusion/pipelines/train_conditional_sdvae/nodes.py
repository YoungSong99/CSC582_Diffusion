from pokemon_diffusion.training.datamodule import create_dataloaders
from pokemon_diffusion.training.train_conditional_sdvae import train_conditional_sdvae_model

def train_conditional_sdvae_node(train_metadata, val_metadata, params):
    train_loader, val_loader = create_dataloaders(
        train_metadata=train_metadata,
        val_metadata=val_metadata,
        img_size=params["data"]["img_size"],
        batch_size=params["data"]["batch_size"],
        normalization_range=params["data"]["normalization_range"],
        num_workers=params["data"].get("num_workers", 2),
    )

    model, history, result_img = train_conditional_sdvae_model(
        train_loader=train_loader,
        val_loader=val_loader,
        params=params,
    )

    return history, result_img