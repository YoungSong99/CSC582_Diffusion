import kagglehub
from pathlib import Path
import shutil

datasets = {
    "pokemon_sprites_csv": {
        "slug": "yehongjiang/pokemon-sprites-images"
    },
    "pokemon_gen_one_images": {
        "slug": "thedagger/pokemon-generation-one"
    },
    "pokemon_library_images": {
        "slug": "divyanshusingh369/complete-pokemon-library-32k-images-and-csv"
    }
}

def download_datasets(datasets):
    downloaded = {}

    for name, info in datasets.items():
        print(f"Downloading {name}...")

        path = kagglehub.dataset_download(info["slug"])

        save_dir = Path(f"data/01_raw/{name}")
        save_dir.mkdir(parents=True, exist_ok=True)

        if not any(save_dir.iterdir()):
            shutil.copytree(path, save_dir, dirs_exist_ok=True)

        downloaded[name] = str(save_dir)

        print(f"Saved to: {save_dir}")

    return downloaded


download_datasets(datasets)