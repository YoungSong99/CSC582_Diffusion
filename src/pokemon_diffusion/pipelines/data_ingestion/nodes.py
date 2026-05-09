import re
import pandas as pd
from pathlib import Path
import pandas as pd

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def normalize_node(pokedex: pd.DataFrame, db: pd.DataFrame):
    pokedex["_name"] = pokedex["name"].str.strip().str.lower()
    db["_name"] = (
        db["Pokemon"]
            .str.strip()
            .str.lower()
            .str.replace(r"\s+forme$", "", regex=True)   
            .str.replace(r"%", "", regex=False)          
            .str.strip()
        )

    return pokedex, db


def norm_db(name: str) -> str:

    if pd.isna(name):
        return ""

    # unicode normalize
    name = (
        name.replace("é", "e")
        .replace("♀", " female")
        .replace("♂", " male")
    )

    # remove parentheses
    name = re.sub(r"\(.*?\)", "", name)
    name = name.lower().strip()

    # keep only alnum + space
    name = re.sub(r"[^a-z0-9 ]", "", name)

    # prefix → suffix
    prefixes = [
        ("mega ", " mega"),
        ("galarian ", " galar"),
        ("hisuian ", " hisui"),
        ("alolan ", " alola"),
        ("primal ", " primal"),
        ("own tempo ", " own tempo"),
        ("heat ", " heat"),
        ("wash ", " wash"),
        ("frost ", " frost"),
        ("fan ", " fan"),
        ("mow ", " mow"),
        ("white ", " white"),
        ("black ", " black"),
        ("ash ", " ash"),
        ("dusk mane ", " dusk mane"),
        ("dawn wings ", " dawn wings"),
        ("ultra ", " ultra"),
    ]

    for prefix, suffix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):] + suffix
            break

    # remove noisy words
    remove_words = [
        " forme",
        " form",
        " style",
        " mode",
        " face",
        " cloak",
        " size",
        " hero of many battles",
        " crowned sword",
        " crowned shield",
    ]

    for word in remove_words:
        name = name.replace(word, "")

    # special cases
    name = (
        name.replace(" crowned sword", " crowned")
        .replace(" crowned shield", " crowned")
    )

    return re.sub(r"\s+", " ", name).strip()


def norm_dex(name: str) -> str:

    if pd.isna(name):
        return ""

    name = name.replace("é", "e").lower().strip()

    name = re.sub(r"[^a-z0-9 ]", "", name)

    name = (
        name.replace(" totem", "")
        .replace(" gmax", "")
        .replace(" eternamax", "")
    )

    name = re.sub(r"nidoran f$", "nidoran female", name)
    name = re.sub(r"nidoran m$", "nidoran male", name)

    return re.sub(r"\s+", " ", name).strip()


def merge_node(pokedex, db):
    pokedex = pokedex.copy()
    db = db.copy()

    pokedex["_key"] = pokedex["name"].apply(norm_dex)
    db["_key"] = db["_name"].apply(norm_db)

    merged = pokedex.merge(
        db,
        on="_key",
        how="left",
    )

    return merged


def clean_node(df: pd.DataFrame, cols_to_keep: list, cat_cols: list) -> pd.DataFrame:


    df.columns = (
        df.columns
        .str.strip()              
        .str.lower()             
        .str.replace(' ', '_')   
        .str.replace(r'[^a-z0-9_]', '', regex=True)
    )

    df = df[cols_to_keep].copy()

    encoders = {}

    for col in cat_cols:
        if col not in df:
            continue

        df[col] = df[col].fillna("None")
        categories = sorted(df[col].unique())

        enc = {c: i for i, c in enumerate(categories)}
        df[col + "_id"] = df[col].map(enc)

        encoders[col] = enc

    return df, encoders


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
