import re
import pandas as pd
from rapidfuzz import process, fuzz


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


def clean_node(df: pd.DataFrame, cols_to_keep: list) -> pd.DataFrame:

    df.columns = (
        df.columns
        .str.strip()              
        .str.lower()             
        .str.replace(' ', '_')   
        .str.replace(r'[^a-z0-9_]', '', regex=True)
    )

    df = df[cols_to_keep].copy()

    return df


def encode_node(df: pd.DataFrame, cat_cols: list[str]):
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


