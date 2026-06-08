import pandas as pd
from pathlib import Path
from sklearn.utils import resample
import json

"""
1. Read cleaned_manifest.csv
2. Check image_path exists
3. Create combined_label from type/color/shape
4. Split into train/val/test using combined_label
5. Balance train set only
6. Encode labels into integer IDs
7. Save train/val/test manifests
"""


def check_image_paths(df, path_col="image_path", image_root=None):

    def resolve_path(p):
        p = Path(p)
        if p.is_absolute():
            return p
        return image_root / p if image_root else p

    df["image_path"] = df[path_col].apply(resolve_path)
    df["image_exists"] = df["image_path"].apply(lambda p: p.exists())

    return df


def create_combined_label(df, label_cols):
    df = df.copy()
    df["combined_label"] = df[label_cols].astype(str).agg("_".join, axis=1)
    return df


def encode_labels_before_split(df, label_cols, save_dir=None):
    df = df.copy()
    mappings = {}

    for col in label_cols:
        classes = sorted(df[col].dropna().unique())
        label_to_id = {label: idx for idx, label in enumerate(classes)}
        mappings[col] = label_to_id
        df[f"{col}_id"] = df[col].map(label_to_id).astype(int)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "label_mappings.json", "w") as f:
            json.dump(mappings, f, indent=2)

    return df, mappings


def split_dataframe(df, label_col="combined_label", train_size=0.7, val_size=0.15, test_size=0.15, seed=42):
    df = df.copy()
    train_dfs = []
    val_dfs = []
    test_dfs = []

    for label in df[label_col].unique():
        label_df = df[df[label_col] == label]
        train_df = label_df.sample(frac=train_size, random_state=seed)
        remaining_df = label_df.drop(train_df.index)
        val_df = remaining_df.sample(frac=val_size / (val_size + test_size), random_state=seed)
        test_df = remaining_df.drop(val_df.index)

        train_dfs.append(train_df)
        val_dfs.append(val_df)
        test_dfs.append(test_df)

    return pd.concat(train_dfs), pd.concat(val_dfs), pd.concat(test_dfs)


def balance_dataframe(dataframe,  label_col="combined_label", max_multiplier=3, max_samples_per_class=300, seed=42):
    df = dataframe.copy()
    counts = df[label_col].value_counts()
    balanced_dfs = []

    for label in counts.index:
        label_df = df[df[label_col] == label]
        target_size = min(counts.max(), len(label_df) * max_multiplier, max_samples_per_class)
        balanced_label_df = resample(label_df, replace=True, n_samples=target_size, random_state=seed)
        balanced_dfs.append(balanced_label_df)

    balanced_dataframe = pd.concat(balanced_dfs)
    balanced_dataframe = balanced_dataframe.sample(frac=1, random_state=seed).reset_index(drop=True)

    return balanced_dataframe

