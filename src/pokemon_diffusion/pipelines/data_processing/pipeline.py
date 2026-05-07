from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    normalize_node,
    merge_node,
    clean_node,
    encode_node,
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
                inputs=["pokedex_norm", "db_norm", "params:dataset.fuzzy_cutoff"],
                outputs="merged",
                name="merge_node",
            ),

            node(
                func=clean_node,
                inputs=["merged", "params:dataset.cols_to_keep"],
                outputs="cleaned",
                name="clean_node",
            ),

            node(
                func=encode_node,
                inputs=["cleaned", "params:dataset.cat_cols"],
                outputs=["cleaned_dataset", "encoders"],
                name="encode_node",
            ),
        ]
    )