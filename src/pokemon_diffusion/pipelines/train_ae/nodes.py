from pokemon_diffusion.models.ae import AutoEncoder
from pokemon_diffusion.training.datamodule import create_dataloaders
from pokemon_diffusion.training.train_ae import train_ae_model

def train_ae_node(train_metadata, val_metadata, params):
    train_loader, val_loader = create_dataloaders(
        train_metadata=train_metadata,
        val_metadata=val_metadata,
        img_size=params["data"]["img_size"],
        batch_size=params["data"]["batch_size"],
        normalization_range=params["data"]["normalization_range"],
        num_workers=params["data"].get("num_workers", 2),
    )

    model = AutoEncoder(
        img_size=params["data"]["img_size"],
        latent_dim=params["model"]["latent_dim"],
        activation=params["model"]["activation"],
        norm_type=params["model"]["norm_type"],
        output_activation=params["model"]["output_activation"],
        base_channels=params["model"]["base_channels"],
    )

    history, result_img = train_ae_model(model=model, train_loader=train_loader, val_loader=val_loader, params=params,)

    return history, result_img