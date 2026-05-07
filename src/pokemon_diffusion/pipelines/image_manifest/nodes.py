from pathlib import Path


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


def merge_paths(row):
    sprite = row["sprite_paths"] if isinstance(row["sprite_paths"], list) else []
    library = row["library_paths"] if isinstance(row["library_paths"], list) else []
    return sprite + library


def image_manifest_node(df, sprites_root: str, library_roots: list[str]):

    sprite_index = build_sprite_index(sprites_root)
    library_index = build_library_index(library_roots)

    def get_sprite_path(key):
        return sprite_index.get(key, [])

    def get_library_path(key):
        return library_index.get(key, [])

    df["sprite_paths"] = df["_key"].apply(get_sprite_path)
    df["library_paths"] = df["_key"].apply(get_library_path)

    df["image_paths"] = df.apply(merge_paths, axis=1)

    return df