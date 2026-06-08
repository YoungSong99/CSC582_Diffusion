from kedro.pipeline import node, pipeline
from .nodes import (
    check_image_paths,
    create_combined_label,
    encode_labels_before_split,
    split_dataframe,
    balance_dataframe,
)

"""
data/05_model_input/train_metadata.csv
data/05_model_input/val_metadata.csv
data/05_model_input/test_metadata.csv
data/04_feature/label_encoders/label_mappings.json
"""

def create_pipeline(**kwargs):
    return pipeline([
        node(
            func=check_image_paths,
            inputs=["cleaned_metadata", "params:path_col", "params:image_root"],
            outputs="checked_metadata",
            name="check_image_paths_node",
        ),
        node(
            func=create_combined_label,
            inputs=["checked_metadata", "params:label_cols"],
            outputs="metadata_with_combined_label",
            name="create_combined_label_node",
        ),
        node(
            func=encode_labels_before_split,
            inputs=[
                "metadata_with_combined_label",
                "params:label_cols",
                "params:label_mapping_save_dir",
            ],
            outputs=["encoded_metadata", "label_mappings"],
            name="encode_labels_node",
        ),
        node(
            func=split_dataframe,
            inputs=[
                "encoded_metadata",
                "params:label_col",
                "params:train_size",
                "params:val_size",
                "params:test_size",
                "params:seed",
            ],
            outputs=["train_metadata_raw", "val_metadata", "test_metadata"],
            name="split_dataframe_node",
        ),
        node(
            func=balance_dataframe,
            inputs=[
                "train_metadata_raw",
                "params:label_col",
                "params:max_multiplier",
                "params:max_samples_per_class",
                "params:seed",
            ],
            outputs="train_metadata",
            name="balance_train_dataframe_node",
        ),
    ])