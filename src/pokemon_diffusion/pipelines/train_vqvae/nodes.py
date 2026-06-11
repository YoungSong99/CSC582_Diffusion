from pokemon_diffusion.models.vqvae import VQVAE
from pokemon_diffusion.training.datamodule import create_dataloaders
from pokemon_diffusion.training.train_vqvae import train_vqvae_model

def train_vqvae_node(train_metadata, val_metadata, params):
    train_loader, val_loader = create_dataloaders(
        train_metadata=train_metadata,
        val_metadata=val_metadata,
        img_size=params["data"]["img_size"],
        batch_size=params["data"]["batch_size"],
        normalization_range=params["data"]["normalization_range"],
        num_workers=params["data"].get("num_workers", 2),
    )
    
    model = VQVAE(
    img_size=params["data"]["img_size"],
    latent_channels=params["model"].get("latent_channels", 256),
    num_embeddings=params["model"].get("num_embeddings", 512),
    activation=params["model"]["activation"],
    norm_type=params["model"]["norm_type"],
    output_activation=params["model"]["output_activation"],
    base_channels=params["model"]["base_channels"],
    commitment_cost=params["training"].get("commitment_cost", 0.25),
)

    history, result_img = train_vqvae_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        params=params,
    )

    return history, result_img