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


def remove_name_adj(name):
    if pd.isna(name):
        return ""
    
    name = str(name).lower()
    
    stopwords = [
        'mega', 'galarian', 'alolan', 'hisuian', 'paldean', 
        'standard', 'mode', 'zen', 'normal', 'attack', 'defense', 
        'origin', 'therian', 'crowned', 'shield', 'sword', 'form', 'forme',
        'two-segment', 'ice face', 'zen mode', 'gigantamax'
    ]
    
    pattern = r'\b(' + '|'.join(stopwords) + r')\b'
    name = re.sub(pattern, '', name)
    
    name = re.sub(r'[^a-z0-9]', ' ', name) 
    name = ' '.join(name.split())
    
    return name


def merge_node(pokedex, db, cutoff=85):

    merged = db.merge(pokedex, on="_name", how="left")

    merged["_name_clean"] = merged["_name"].apply(remove_name_adj)

    ref = merged.dropna(subset=["type1", "primary_color", "shape"])
    ref_names = ref["name"].unique().tolist()

    mask = merged[["type1", "primary_color", "shape"]].isna().any(axis=1)

    for idx in merged[mask].index:

        q = merged.at[idx, "_name_clean"]
        if not q:
            continue

        match = process.extractOne(
            q,
            ref_names,
            scorer=fuzz.token_sort_ratio
        )

        if not match:
            continue

        best, score, _ = match

        if score >= cutoff:

            row = ref[ref["name"] == best].iloc[0]

            merged.at[idx, "name"] = row["name"]
            merged.at[idx, "type1"] = row["type1"]
            merged.at[idx, "type2"] = row["type2"]
            merged.at[idx, "primary_color"] = row["primary_color"]
            merged.at[idx, "shape"] = row["shape"]

    return merged


def clean_node(df: pd.DataFrame, cols_to_keep: list) -> pd.DataFrame:

    df = df[cols_to_keep].copy()

    df.columns = (
        df.columns
        .str.strip()              
        .str.lower()             
        .str.replace(' ', '_')   
        .str.replace(r'[^a-z0-9_]', '', regex=True)
    )

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


