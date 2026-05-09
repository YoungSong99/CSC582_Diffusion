from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    normalize_node,
    merge_node,
    clean_node,
    image_manifest_node
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=normalize_node,
                inputs=["pokedex", "pokemon_db"],
                outputs=["pokedex_norm", "db_norm"],
                name="normalize_node",
            ),

            node(
                func=merge_node,
                inputs=["pokedex_norm", "db_norm"],
                outputs= "merged",
                name="merge_node",
            ),

            node(
                func=clean_node,
                inputs=["merged", 
                        "params:dataset.cols_to_keep",
                        "params:dataset.cat_cols"],
                outputs=["cleaned", "label_encoders"],
                name="clean_node",
            ),

            node(
                func=image_manifest_node,
                inputs=[
                    "cleaned",
                    "params:sprites_root",
                    "params:library_roots",
                ],
                outputs= "merged_dataset",
                name="image_manifest_node",
            ),
        ]
    )