from pathlib import Path
import pandas as pd
import ast


IMG_EXTS = {".png", ".jpg", ".jpeg"}


def build_sprite_index(sprites_root: str):
    sprites_root = Path(sprites_root)

    index = {}

    for folder in sprites_root.iterdir():
        if not folder.is_dir():
            continue

        name = folder.name.lower().split("-")[1]

        files = [
                str(p)
                for p in folder.rglob("*")
                if p.suffix.lower() in IMG_EXTS
            ]

        if name not in index:
            index[name] = []

        index[name].extend(files)
        print(index)

        break

    return index

def build_library_index(library_roots: list[str]):
    index = {}

    for root in library_roots:
        root = Path(root)

        for folder in root.iterdir():
            if not folder.is_dir():
                continue

            name = folder.name.lower()

            files = [
                str(p)
                for p in folder.rglob("*")
                if p.suffix.lower() in IMG_EXTS
            ]

            if name not in index:
                index[name] = []

            index[name].extend(files)

    return index


def build_image_manifest(df: pd.DataFrame, sprites_root: str, library_roots: list[str],):

    sprite_index = build_sprite_index(sprites_root)
    library_index = build_library_index(library_roots)


    def resolve_sprite(row):
        try:
            fn = ast.literal_eval(row["image_fn"])[0]
            key = (
                f"{int(row.name):04d}",
                str(row["name"]).lower(),
                str(row["pokedex_id"]),
                fn
            )
            return sprite_index.get(key, None)
        except:
            return None

    df["sprite_path"] = df.apply(resolve_sprite, axis=1)

    def clean_name(name):
        return str(name).split(" ")[0].lower()

    df["base_name"] = df["name"].apply(clean_name)
    df["library_paths"] = df["base_name"].map(library_index)

    df["library_paths"] = df["library_paths"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    print("sprite:", df["sprite_path"].notna().sum())
    print("library:", df["library_paths"].apply(len).gt(0).sum())

    return df