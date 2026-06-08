from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class PokemonSpriteDataset(Dataset):
    def __init__(self, dataframe, transform=None, return_path=False):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        sample = {
            "image": image,
            "type_id": int(row["type_id"]),
            "color_id": int(row["primary_color_id"]),
            "shape_id": int(row["shape_id"]),
        }

        if self.return_path:
            sample["image_path"] = str(image_path)

        return sample