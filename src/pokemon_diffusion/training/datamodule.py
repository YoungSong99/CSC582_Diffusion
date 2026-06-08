import torch
from torch.utils.data import DataLoader
from pokemon_diffusion.datasets.pokemon_dataset import PokemonSpriteDataset
from pokemon_diffusion.datasets.transform import get_train_transform, get_eval_transform

def create_dataloaders(train_metadata, val_metadata, img_size=96, batch_size=32, 
                          normalization_range="minus_one_to_one", root_dir=None, num_workers=2):
    
    train_transform = get_train_transform(img_size=img_size, normalization_range=normalization_range,)
    eval_transform = get_eval_transform(img_size=img_size, normalization_range=normalization_range,)

    train_dataset = PokemonSpriteDataset(dataframe=train_metadata, transform=train_transform,)
    val_dataset = PokemonSpriteDataset(dataframe=val_metadata, transform=eval_transform,)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(),)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(),)

    return train_loader, val_loader